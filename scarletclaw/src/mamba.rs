use crate::tensor::{Tensor, TernaryTensor};

/// Represents a single Structured State Space (SSM) block, the core of Mamba architectures.
/// This replaces traditional Multi-Head Attention by compressing sequence history
/// into a hidden state `h_t` using a discretized state-space equation:
/// h_t = A * h_{t-1} + B * x_t
/// y_t = C * h_t
pub struct SsmBlock {
    // In a pure BitMamba implementation, standard projections are replaced with Ternary weights
    pub proj_in: TernaryTensor,  // (hidden_size, expand_size)
    pub proj_out: TernaryTensor, // (expand_size, hidden_size)

    // SSM specific parameters
    pub x_proj: Tensor, // Linear projection from expand_size -> (dt_rank + state_size * 2)
    pub dt_proj: Tensor, // Linear projection from dt_rank -> expand_size
    pub a_log: Tensor,  // Parameterized A matrix (often stored in log space)
    pub d: Tensor,      // Skip connection parameter

    // Config dims
    pub dt_rank: usize,

    // Internal state carried forward sequentially
    pub h_state: Tensor, // (batch, expand_size, state_size)
}

impl SsmBlock {
    pub fn new(hidden_size: usize, expand_size: usize, state_size: usize) -> Self {
        // Mamba typically defines dt_rank as ceil(hidden_size / 16)
        let dt_rank = (hidden_size as f32 / 16.0).ceil() as usize;

        Self {
            // Dummy initialization for scaffolding
            proj_in: TernaryTensor::new(
                vec![0; hidden_size * expand_size],
                vec![hidden_size, expand_size],
            ),
            proj_out: TernaryTensor::new(
                vec![0; expand_size * hidden_size],
                vec![expand_size, hidden_size],
            ),

            // Linear projection weights for creating B, C, and dt
            x_proj: Tensor::zeros(vec![expand_size, dt_rank + state_size * 2]),
            dt_proj: Tensor::zeros(vec![dt_rank, expand_size]),

            a_log: Tensor::zeros(vec![expand_size, state_size]),
            d: Tensor::zeros(vec![expand_size]),
            dt_rank,
            h_state: Tensor::zeros(vec![1, expand_size, state_size]),
        }
    }

    /// Forward pass for a single timestep (token).
    /// `x` is (1, hidden_size).
    /// Returns (1, hidden_size).
    pub fn forward_step(&mut self, x: &Tensor) -> Tensor {
        // 1. Up-project input using 1.58-bit ternary matmul
        // x_expand = x @ proj_in -> (1, expand_size)
        let x_expand = x.ternary_matmul(&self.proj_in);

        let expand_size = self.h_state.shape[1];
        let state_size = self.h_state.shape[2];

        // 2. Data-dependent parameter projections
        // In Mamba, B, C, and dt are functions of the expanded input.
        // First we project x_expand to a combined vector of size (dt_rank + state_size * 2)
        let ssm_params = x_expand.matmul(&self.x_proj);

        // Slice the combined vector into dt, B, and C
        let mut dt_in = Tensor::zeros(vec![1, self.dt_rank]);
        let mut b = Tensor::zeros(vec![state_size]);
        let mut c = Tensor::zeros(vec![state_size]);

        for i in 0..self.dt_rank {
            dt_in.data[i] = ssm_params.data[i];
        }
        for i in 0..state_size {
            b.data[i] = ssm_params.data[self.dt_rank + i];
            c.data[i] = ssm_params.data[self.dt_rank + state_size + i];
        }

        // Project dt_in up to expand_size to get the actual step sizes
        let mut dt = dt_in.matmul(&self.dt_proj);
        // Softplus is usually applied to dt here for stability
        for i in 0..expand_size {
            let val = dt.data[i];
            dt.data[i] = if val > 0.0 {
                val + (-val).exp().ln_1p()
            } else {
                val.exp().ln_1p()
            };
        }

        // 3. Discretization and State Update
        // A_bar = exp(dt * A)
        // B_bar = (dt * B)
        // h_t = A_bar * h_{t-1} + B_bar * x_t
        let mut y_out = Tensor::zeros(vec![1, expand_size]);

        use rayon::prelude::*;

        // We parallelize over the `expand_size` dimension because each scalar channel
        // in the SSM operates entirely independently of the others.
        // We zip the output array, the input slices, and the state chunks to mutate safely.
        y_out.data.par_iter_mut()
            .zip(dt.data.par_iter())
            .zip(x_expand.data.par_iter())
            .zip(self.h_state.data.par_chunks_mut(state_size))
            .zip(self.a_log.data.par_chunks(state_size))
            .zip(self.d.data.par_iter())
            .for_each(|(((((y_val_out, &dt_val), &x_val), h_chunk), a_chunk), &d_val)| {

                let mut y_val = 0.0;

                for s in 0..state_size {
                    let a_val = -a_chunk[s].exp(); // Enforce negative A
                    let a_bar = (dt_val * a_val).exp();
                    let b_bar = dt_val * b.data[s];

                    let prev_h = h_chunk[s];

                    // Update internal hidden state
                    let new_h = a_bar * prev_h + b_bar * x_val;
                    h_chunk[s] = new_h;

                    // Compute output slice
                    y_val += new_h * c.data[s];
                }

                // Add skip connection
                y_val += x_val * d_val;
                *y_val_out = y_val;
            });

        // 4. Down-project back to hidden size using Ternary weights
        // y = y_out @ proj_out -> (1, hidden_size)
        y_out.ternary_matmul(&self.proj_out)
    }

    /// Resets the internal state (required when starting a new sequence).
    pub fn reset_state(&mut self) {
        let size = self.h_state.data.len();
        self.h_state.data = vec![0.0; size];
    }
}
