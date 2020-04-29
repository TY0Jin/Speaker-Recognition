DATASET_DIR = 'C:/Users/Jin/Desktop/wav'

BATCH_NUM_TRIPLETS = 20  # should be a multiple of 3

# very dumb values. I selected them to have a blazing fast training.
# we will change them to their true values (to be defined?) later.
NUM_FRAMES = 2
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.5,1.5)  # (start_sec, end_sec)

CHECKPOINT_FOLDER = 'D:/Graduation_project/Code/deep_speaker_standard/checkpoints'
LOSS_FILE = CHECKPOINT_FOLDER + '/losses.txt'
