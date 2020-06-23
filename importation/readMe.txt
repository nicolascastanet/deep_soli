exemple d'appel du script:

python importation/main.py --data_path data --nb_frames 40 --mode cross_user --channel 4

nb_frames : nombre de frames des séquences lors du subsampling (on ajoute des frames nuls ou on supprime des frames pour ramener toutes les séquences à la même taille)
nb_channel : 4 pour Soli
mode : mode 'cross user' -> 11x25x10 = 2750 sequences (10 users) utilisé dans le papier Soli
       mode 'single user' -> 11x50x5 = 2750 sequences (1 users) utilisé pour la reconnaissance personalisée


output : fichier pickle

	import pickle
	soli_data = pickle.load( open( "soli_numpy.p", "rb" ) )

dim = (nb_geste, nb_sessions, nb_instances, nb_channel, nb_frames, 32, 32)