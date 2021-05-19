import glob
import torch
import tqdm
# change all notes with values -1, 1 -> 0

paths = glob.glob(f"./data/VocalSet/cache/*.pt")
with tqdm.tqdm(paths) as t:
    t.set_description("Verifying data integrity")
    for path in t:
        data = torch.load(path)
        save = False
        for note in data.notes:
            if note.pitch == -1 or note.pitch == 1:
                note.pitch = 0
                save = True
        if save:
            t.set_description(f'Fixing file {path}')
            torch.save(data, path)