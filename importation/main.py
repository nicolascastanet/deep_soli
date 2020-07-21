import argparse
from soli import Soli
import pickle
import numpy as np


def main(path, nb_frames = 40, mode = 'cross_user', num_channel=4):

    s = Soli(mode, nb_frames, num_channel)
    s.load_data(path)

    print("loading data ...")

    np.save('soli_data.npy',s.data)
    np.save('soli_gestLabels.npy', s.gestureLabels)
    np.save('soli_framesLabels.npy', s.frameLabels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pickle")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--nb_frames', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--channel', type=int)

    args = parser.parse_args()

    main(args.data_path, int(args.nb_frames), args.mode, int(args.channel))