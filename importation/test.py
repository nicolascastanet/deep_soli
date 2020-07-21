import argparse
parser = argparse.ArgumentParser(description="Generate pickle")
parser.add_argument('--data_path', type=str)
parser.add_argument('--nb_frames', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--channel', type=int)

args = parser.parse_args()


print(args.mode)