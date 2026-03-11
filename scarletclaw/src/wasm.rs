use anyhow::{Result, bail};
use wasmtime::*;

/// Represents an execution environment capable of safely running dynamically
/// loaded WASM tools. This bridges our `SandboxConfig` constraints into a
/// rigorous virtual machine.
pub struct WasmSandbox {
    engine: Engine,
}

impl WasmSandbox {
    pub fn new() -> Result<Self> {
        let mut config = Config::new();
        config.wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Enable);

        // Critical safeguard: Enable execution bounds to prevent infinite loops.
        config.consume_fuel(true);

        let engine = Engine::new(&config)?;
        Ok(Self { engine })
    }

    /// Compiles and executes a raw WASM module's `run` function safely.
    pub fn execute_wasm_module(&self, wasm_bytes: &[u8], args: &str) -> Result<String> {
        let mut store = Store::new(&self.engine, ());

        // Inject a maximum fuel limit so the script cannot infinite loop.
        store.set_fuel(1_000_000)?;

        let module = match Module::new(&self.engine, wasm_bytes) {
            Ok(m) => m,
            Err(e) => bail!("Failed to compile WASM module: {}", e),
        };

        let instance = match Instance::new(&mut store, &module, &[]) {
            Ok(i) => i,
            Err(e) => bail!("Failed to instantiate WASM module: {}", e),
        };

        // Locate and invoke the "run" function the tool automatically wraps the code in.
        let run_func = instance.get_typed_func::<(), ()>(&mut store, "run")?;

        match run_func.call(&mut store, ()) {
            Ok(_) => Ok(format!("Executed WASM module successfully with args: {}", args)),
            Err(e) => bail!("WASM Execution trap/error: {}", e),
        }
    }
}
