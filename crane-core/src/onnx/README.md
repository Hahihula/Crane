Crane-maintained ONNX evaluator sources.

`eval.rs` and `onnx.proto3` are kept here so Crane can independently add
ONNX operators and parser fixes as models require them.

note: this is the best way since candle official merge PR so slow.
We welcome any third-part op support PR inside Crane for any model.
