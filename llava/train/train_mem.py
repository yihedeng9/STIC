from llava.train.train_ori import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
