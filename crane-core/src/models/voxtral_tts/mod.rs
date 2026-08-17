pub mod codec;
pub mod model;
pub mod modeling;

pub use codec::CodecDecoder;
pub use model::{
    BEGIN_AUDIO, BOS, END_AUDIO_CODE, INST, INST_END, Model, PromptSegment, SpeechStream,
    VoxtralConfig, build_prompt_segments, load_tokenizer, load_voice_embedding,
};
pub use modeling::{AcousticTransformer, AudioCodebookEmbedding, VoxtralLlm};

#[allow(unused_imports)] // used by crane-serve later
pub(crate) use model::{
    AcousticTransformerArgs, AudioModelArgs, AudioTokenizerArgs, MultimodalConfig,
};
#[allow(unused_imports)] // used by crane-serve later
pub(crate) use modeling::rename_voxtral_transformer_keys;
