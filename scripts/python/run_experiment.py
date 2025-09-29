import subprocess

def run_train_commands():
    base_command = "python src/train.py --multirun --config-dir=./config --config-name=train_slurm.yaml"
    decoder_layers = [2]
    hidden_sizes = [512,768,1024]
    full_commands = []
    # for i, decoder_layer in enumerate(decoder_layers):
    #     full_command = base_command + f" model.decoder.decoder_layers={decoder_layer} &"
    #     full_commands.append(full_command)

    for i, hidden_size in enumerate(hidden_sizes):
        full_command = base_command + f" model.decoder.tie_word_embeddings=true model.encoder.hidden_size={hidden_size} &"
        full_commands.append(full_command)

    for i, full_command in enumerate(full_commands, 1):
        print(f"启动命令 {i}/{len(full_commands)}: {full_command}")
        subprocess.Popen(full_command, shell=True)

if __name__ == "__main__":
    run_train_commands()