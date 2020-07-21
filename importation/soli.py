import os, re
import numpy as np
import h5py


class Soli:
    def __init__(self, mode = 'cross_user', nb_frames = 40, channels = 4):
        
        self.nb_frames = nb_frames
        self.num_channel = channels
        self.mode = mode
        if self.mode == 'cross_user':
            self.sessions = [2,3,5,6,8,9,10,11,12,13]
        elif self.mode == 'single_user':
            self.sessions = [0,1,4,7,13,14]

    # get files in label directories
    def get_h5(self, file_dir):
        return [os.path.join(file_dir, f)
            for f in os.listdir(file_dir)
            if os.path.isfile(os.path.join(file_dir, f)) and 'h5' in f]

    # get list of target image dir names
    def get_dir(self, file_dir):
        return [os.path.join(file_dir, d)
            for d in os.listdir(file_dir)
            if os.path.isdir(os.path.join(file_dir, d))]

    def load_data(self, data_path):
        #numpy init
        if self.mode == 'cross_user':
            #nb_gestures*nb_sessions*nb_instances*(nb_channels*nb_frames*32*32)
            self.data = np.zeros((11,10,25,self.num_channel,self.nb_frames,32,32))
            self.frameLabels = np.zeros((11,10,25,self.nb_frames)) # One label per frame
            self.gestureLabels = np.zeros((11,10,25)) #One label per gesture
            source_files = self.get_h5(data_path) #get gestures files
            
        #get gestures files
        source_files = self.get_h5(data_path)

        for s in source_files:
            
            d = os.path.basename(s).split('_')
            gestureID, sessionID, inst = int(d[0]), int(d[1]), int(d[2].split('.')[0])
            
            if sessionID in set(self.sessions):
                print('Generating', s)
                with h5py.File(s, 'r') as f:
                    lab = f['label'][()]
                    num_frames = lab.shape[0]
                    self.gestureLabels[gestureID, self.sessions.index(sessionID), inst] = lab[0]

                    for c in range(self.num_channel):
                        # Data and label are numpy arrays
                        data = f['ch{}'.format(c)][()].reshape(num_frames,32,32)
                        
                        #Subsampling
                        if num_frames > self.nb_frames:
                            self.data[gestureID, self.sessions.index(sessionID), inst, c, :,:,:] = data[:self.nb_frames]
                            self.frameLabels[gestureID, self.sessions.index(sessionID), inst, :] = lab[:self.nb_frames].squeeze()
                        else:
                            self.data[gestureID, self.sessions.index(sessionID), inst, c, :num_frames,:,:] = data
                            self.frameLabels[gestureID, self.sessions.index(sessionID), inst, :num_frames] = lab.squeeze()