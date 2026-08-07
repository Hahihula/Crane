fn main() {
    println!("cargo::rerun-if-changed=kernels/cuda/");
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "onnx")]
    {
        println!("cargo::rerun-if-changed=src/onnx/onnx.proto3");
        prost_build::compile_protos(&["src/onnx/onnx.proto3"], &["src/onnx"])
            .expect("failed to generate Crane's vendored ONNX protobuf bindings");
    }

    // Only compile CUDA kernels when the cuda feature is enabled.
    #[cfg(feature = "cuda")]
    {
        use std::env;
        use std::path::PathBuf;

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        let builder = bindgen_cuda::Builder::default()
            .kernel_paths_glob("kernels/cuda/**/*.cu")
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O3");

        let bindings = builder.build_ptx().expect("Failed to compile CUDA kernels");
        bindings
            .write(out_dir.join("crane_kernels_ptx.rs"))
            .expect("Failed to write PTX bindings");
    }
}
