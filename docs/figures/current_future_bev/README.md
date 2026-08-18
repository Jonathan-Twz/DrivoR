# Current + Future BEV DrivoR Figures

This folder contains academic SVG diagrams for the current + privileged future BEV scorer/decoder co-tuning experiment.

- `current_future_bev_framework.svg`: end-to-end framework from camera/lidar/current BEV/future BEV inputs to BEV-aware decoder, scorer, and losses.
- `bev_aware_decoder_scorer_block.svg`: repeated transformer block used by both `BevAwareTrajectoryDecoder` and `BevAwareScorer`.

Implementation note: the current code tokenizes both current BEV and privileged future BEV when enabled, but `DrivoRModel.forward` routes future BEV tokens to the decoder/scorer whenever they are available. Current BEV tokens are used as fallback when future BEV tokens are absent; they are not concatenated with future BEV tokens in the current implementation.
