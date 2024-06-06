import argparse
import yaml

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Override config values from the command line")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file")
    
    # 필요한 경우 여기와 아래에도 추가
    parser.add_argument("--lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size per device for training")
    parser.add_argument("--max_steps", type=int, help="Max steps for training")

    args = parser.parse_args()
    return args

def override_config(config, args):
    if args.lr is not None:
        config['training_args']['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['training_args']['batch_size'] = args.batch_size
    if args.max_steps is not None:
        config['training_args']['max_steps'] = args.max_steps
    return config


if __name__ == "__main__":
    from pprint import pprint
    args = parse_args()
    config = load_config(args.config)
    config = override_config(config, args)

    print(type(config))
    pprint(config)