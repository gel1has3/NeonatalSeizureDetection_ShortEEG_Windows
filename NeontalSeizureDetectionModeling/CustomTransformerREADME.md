1. A["Raw EEG Input"]: dimensions as [batch_size × num_channels × num_timepoints]. 

2. B["Per-Channel Standardization"]:   standardization step: X_hat = (X - mean) / (std + 1e-5).

3. C["Positional Encoding Injection"]: This is the X_tilde = X_hat + self.positional_encoding line. The description "Sinusoidal Channel-Time Encoding".

4. D["Tensor Permutation"]:  "Time → Batch Sequence" is an accurate, higher-level description of X_tilde.permute(2, 0, 1).

5. E["Multi-Head Self-Attention"]:  self.multihead_attn call.

6. F["Attention Output: Dropout + LayerNorm"]:  self.dropout(attn_output) and the subsequent list comprehension with self.norm1.

7. G["Feed-Forward Network"]: s the self.ffn sequential block.

8. H["Residual Connection + Layer Normalization"]: This is the self.norm2(ff_output + X_ring) line.  It  shows the skip connection (F -->|Skip Connection| H) that carries the output of the attention block to be added to the output of the feed-forward network. This is a key feature of the transformer architecture that the code implements.

9. I["Feature Flattening"]: This matches the O.view(O.size(0), -1) operation, preparing the data for the final classification layer.

10. J["Classification Head"]: This is the self.classifier linear layer.

11. K["Prediction Output"]: The final output returned by the function. The  "logits" is an accurate description of the raw outputs from the linear layer before a potential softmax function is applied.
