/// A strictly zero-dependency, bare-metal 1D/2D Tensor implementation
/// focused on foundational mathematics and physics concepts for ML.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// A tensor optimized for 1.58-bit models (like BitNet/BitMamba), where weights
/// are strictly -1, 0, or 1. Represented as `i8` for memory density.
#[derive(Debug, Clone)]
pub struct TernaryTensor {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
}

impl TernaryTensor {
    /// Creates a new ternary tensor.
    pub fn new(data: Vec<i8>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Data length must match shape product");
        // Verify all elements are -1, 0, or 1
        for &val in &data {
            assert!(val >= -1 && val <= 1, "Ternary weights must be -1, 0, or 1");
        }
        Self { data, shape }
    }
}

impl Tensor {
    /// Creates a new tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Creates a new tensor from raw data and a shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Data length must match shape product");
        Self { data, shape }
    }

    /// Element-wise addition. Mutates `self` in place.
    pub fn add_assign(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "Shapes must match for addition");
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }

    /// Element-wise multiplication. Mutates `self` in place.
    pub fn mul_assign(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "Shapes must match for multiplication");
        for i in 0..self.data.len() {
            self.data[i] *= other.data[i];
        }
    }

    /// Computes the dot product of two 1D vectors.
    pub fn dot(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.shape.len(), 1, "Dot product requires 1D tensors");
        assert_eq!(a.shape, b.shape, "Shapes must match for dot product");

        let mut sum = 0.0;
        for i in 0..a.data.len() {
            sum += a.data[i] * b.data[i];
        }
        sum
    }

    /// Performs Matrix Multiplication (MatMul) with a standard floating point tensor.
    /// `self` is (M, K), `other` is (K, N). Returns a new Tensor of (M, N).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "self must be 2D");
        assert_eq!(other.shape.len(), 2, "other must be 2D");

        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(k, other.shape[0], "Inner dimensions must match (M, K) x (K, N)");
        let n = other.shape[1];

        let mut result = Tensor::zeros(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result.data[i * n + j] = sum;
            }
        }

        result
    }

    /// Performs 1.58-bit optimized Matrix Multiplication.
    /// In BitNet/BitMamba architectures, weights are strictly (-1, 0, 1).
    /// This allows us to completely skip floating-point multiplications and use only
    /// additions and subtractions, drastically reducing computational overhead.
    /// `self` is (M, K) representing activations. `weights` is (K, N) representing ternary weights.
    pub fn ternary_matmul(&self, weights: &TernaryTensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "self must be 2D");
        assert_eq!(weights.shape.len(), 2, "weights must be 2D");

        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(k, weights.shape[0], "Inner dimensions must match (M, K) x (K, N)");
        let n = weights.shape[1];

        let mut result = Tensor::zeros(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    let a_val = self.data[i * k + l];
                    let w_val = weights.data[l * n + j];

                    // The core BitNet / 1.58-bit performance hack:
                    // Since w_val is only -1, 0, or 1, we avoid float multiplication entirely.
                    match w_val {
                        1 => sum += a_val,
                        -1 => sum -= a_val,
                        0 => {}, // Skip 0
                        _ => unreachable!("Ternary weights must be -1, 0, 1"),
                    }
                }
                result.data[i * n + j] = sum;
            }
        }

        result
    }

    /// Applies the Softmax activation function over a 1D tensor.
    pub fn softmax(&mut self) {
        assert_eq!(self.shape.len(), 1, "Softmax implemented for 1D tensors here");

        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for &val in &self.data {
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum_exp = 0.0;
        for i in 0..self.data.len() {
            self.data[i] = (self.data[i] - max_val).exp();
            sum_exp += self.data[i];
        }

        for i in 0..self.data.len() {
            self.data[i] /= sum_exp;
        }
    }

    /// Applies the SiLU (Swish) activation function: x * sigmoid(x)
    /// Used heavily in modern physics-inspired or LLaMA/Mamba-style models.
    pub fn silu(&mut self) {
        for i in 0..self.data.len() {
            let x = self.data[i];
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            self.data[i] = x * sigmoid;
        }
    }
}
