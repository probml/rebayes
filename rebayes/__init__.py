import jax

# Add matmul precision to avoid matmul precision error
jax.config.update("jax_default_matmul_precision", "float32")
