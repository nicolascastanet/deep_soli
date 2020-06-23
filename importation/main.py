import argparse
from soli import Soli
import pickle


def main(path, nb_frames = 40, mode = 'cross_user', num_channel=4):

    s = Soli(mode, nb_frames, num_channel)
    s.load_data(path)

    print("loading pickle file ...")
    pickle.dump(s.data, open( "soli_numpy.p", "wb" ) )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pickle")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--nb_frames', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--channel', type=int)

    args = parser.parse_args()

    main(args.data_path, int(args.nb_frames), args.mode, int(args.channel))