import sys
import os


def print_progress(current, total, message=""):
    progress = current * 1. / total
    bar_length = 30
    block = int(round(bar_length * progress))
    sys.stdout.write("\r%s: [%s] - [%5.2f%% - %d/%d]" % (message,
                                                         "=" * block + "-" * (bar_length - block),
                                                         progress * 100,
                                                         current, total,))
    sys.stdout.flush()


def save_model(saver, session, file_name):
    full_dir = os.path.join(os.getcwd(), file_name)
    folder_dir = os.path.dirname(full_dir)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    saver.save(session, full_dir)

