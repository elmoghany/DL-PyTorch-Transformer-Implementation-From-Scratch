# PyTorch Transformer Implementation From Scratch
![hidden](https://media.licdn.com/dms/image/v2/D4D12AQHwm99SQx-EGg/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1680464862455?e=2147483647&v=beta&t=o-jqYBJ4bfOh_Hhq0wIfoPnmiEVhvDXvKcSx69mK2m0)

# Testing the Transformer Model
```
if __name__ == "__main__":
    batch_size = 32
    seq_len = 20
    vocab_size = 1000
    d_model = 512
    num_layers = 6
    heads = 8
    d_ff = 1024
    dropout = 0.3
    
    src = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    trgt = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    print(f"Source input shape: {src.shape}, dtype: {src.dtype}")
    print(f"Target input shape: {trgt.shape}, dtype: {trgt.dtype}")
    
    
    transformer = build_transformer(
        vocab_size=vocab_size,
        sequence_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        heads=heads,
        d_ff=d_ff,
        dropout=dropout
    )

    # Create masks
    src_mask = None  # Set to None for the encoder
    trgt_mask = create_mask(seq_len)
    
    # Forward pass
    try:
        output = transformer(src, trgt, src_mask, trgt_mask)
        print(f"Transformer output shape: {output.shape}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
```
