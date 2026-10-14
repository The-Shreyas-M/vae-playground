# VAE Playground — Draw Digits AND Letters With 2 Sliders 🤖✏️

> **In one line:** This app teaches a computer to doodle handwritten digits (0-9) AND capital letters (A-Z), then lets YOU control its imagination with just 2 sliders.

No AI background needed. If you can move a slider, you can use this project.

---

## What does this actually do?

Imagine every handwritten digit lives on a giant map:

- All the `0`s hang out in one neighbourhood, all the `1`s in another, and so on.
- That map has only **2 directions**: left-right (called `x`) and up-down (called `y`).
- This project **learns that map by itself** by looking at 60,000 photos of digits (the famous MNIST dataset).
- Then it gives you **2 sliders** — one for `x`, one for `y` — and draws whatever digit lives at that spot **instantly**.

Move the sliders slowly and you will literally watch a `3` melt into an `8`. That is a **Variational Autoencoder (VAE)** doing its magic.

### The 30-second picture

```
  You move sliders (x, y)
         |
         v
  +----------------+
  | Decoder brain  |  <-- learned from 60,000 digits
  +----------------+
         |
         v
   28x28 doodle on screen

  Scatter plot = aerial photo of Digit City, coloured by digit 0-9
```

**Jargon buster (only 3 words to know):**

| Word | Dumb-friend meaning |
|------|---------------------|
| `MNIST` | 70,000 tiny photos of handwritten digits. The "practice book" the AI learns from. |
| `VAE` | A student + an artist. The student squeezes a photo into 2 numbers, the artist redraws it from just those 2 numbers. |
| `Latent space` | The secret 2D map where similar digits live close together. Your sliders are the GPS on this map. |

---

## What you will see

1. **Left side — the map:** ~1,000 dots, each dot = one real digit, coloured 0-9. You can see the neighbourhoods.
2. **Right side — 2 sliders:** `Latent x` and `Latent y`. This is your joystick.
3. **Below — the doodle:** The AI's drawing for your current joystick position. Updates live.

---

## Run it in 3 steps (Windows, Mac, Linux)

You need **Python 3.10+** installed. That's it.

### Option A — "I just want to play" (no training, ~1 minute)

The trained brain (`vae_weights.pth`) is already in this repo, so skip training:

```bash
# 1. Make a clean sandbox for this project
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate

# 2. Install the Lego bricks
pip install -r requirements.txt

# 3. Open the playground
streamlit run app.py
```

Your browser opens automatically. Drag the sliders. Done.

> Windows double-click fans: `run.bat` = digits, `runalphabets.bat` = letters (A-Z pre-selected). You can also switch anytime with the **Dataset** buttons at the top of the page.

### Option B — "I want to train it myself"

**Digits (MNIST, ~5 min on CPU):**

```bash
python train.py --dataset mnist
```

**Alphabet (EMNIST Letters A-Z, ~2-4 min quick version on CPU):**

```bash
# Small + fast: 20,000 letters, 3 rounds. Perfect for class demos.
python train.py --dataset emnist-letters --epochs 3 --max-train 20000
```

Want the full alphabet training? Drop `--max-train` (uses all ~125,000 letters, ~10-15 min on CPU):

```bash
python train.py --dataset emnist-letters --epochs 5
```

Other tiny 28×28 datasets work too: `--dataset fashion-mnist`, `--dataset kmnist` (Japanese characters, same size as MNIST).

This will:

- Download the dataset automatically (first time only, into `data/` — this folder is NOT pushed to GitHub)
- Train on whatever you have: NVIDIA GPU → Apple Silicon → plain CPU
- Save 4 files (names depend on dataset, e.g. EMNIST Letters):
  - `vae_emnist_letters_weights.pth` — the trained brain (~3 MB)
  - `latent_emnist_letters.csv` — map coordinates of 1,000 test letters (`x, y, label A-Z`)
  - `latent_emnist_letters_bounds.json` — edges of the map (so sliders line up)
  - `dataset_emnist_letters.json` — which dataset + class names

Then run `streamlit run app.py` as above and pick the dataset from the sidebar — digits and letters can live side by side.

---

## Files — what is what?

| File | What it is, in plain English |
|------|------------------------------|
| `app.py` | The playground website (Streamlit). Sliders + map + doodle. Start here to play. |
| `train.py` | The school. `python train.py --dataset mnist` for digits, `--dataset emnist-letters --epochs 3 --max-train 20000` for a fast A-Z run. |
| `model.py` | The brain blueprint. Squeezer (encoder) + artist (decoder). Only ~100 lines, well commented. |
| `requirements.txt` | Shopping list of Python packages. `pip install -r requirements.txt` buys them all. |
| `vae_weights.pth` | Saved brain after school (digits). Lets you skip training. |
| `latent_data.csv` | 1,000 dots for the digit map: `x, y, digit`. |
| `vae_emnist_letters_weights.pth` | Saved brain trained on the A-Z alphabet (3 epochs, 20k letters). Pick it from the app sidebar. |
| `latent_emnist_letters.csv` | 1,040 dots for the letter map (40 per letter A-Z). |
| `dataset_*.json` / `latent_*_bounds.json` | Sidecars: class names + slider ranges for each dataset. |
| `run.bat` | Double-click launcher for Windows. |

---

## Try these fun experiments (no coding!)

1. **Find digit neighbourhoods:** Set `x ≈ -1, y ≈ +1`. What digit is that? Now slide `x` slowly to `+2`. What does it turn into?
2. **The morph:** Pick a `3`-looking spot, then drag one slider very slowly. Watch it become an `8`, then a `0`. You are walking through Digit City.
3. **The ghost zone:** Go to empty space between colour blobs. The doodle looks blurry / alien? That's the AI guessing a digit that doesn't exist. That's normal — it's dreaming.
4. **Break it (safely):** Drag both sliders to the extreme corners. The drawing gets weird because you left the city. Come back toward `0, 0` and it gets clean again.

---

## How does it work? (60-second version)

1. **Squeezer (Encoder):** `28×28 photo → 2 numbers (x, y)` using 2 convolution layers.
2. **Dice roll (Reparameterisation trick):** Instead of memorising exact spots, it learns a fuzzy cloud per digit. That fuzz is why it can dream up *new* digits, not just copy old ones.
3. **Artist (Decoder):** `2 numbers → 28×28 doodle` using 2 transpose-convolutions.
4. **Report card (Loss):** `How bad is my doodle? (BCE) + How messy is my map? (KL)`. Training = making both small over 5 rounds.

Latent size is fixed at `2` on purpose — so humans can see the whole map on one screen.

---

## Problems? Read this first

| Symptom | Fix |
|---------|-----|
| `vae_weights.pth not found` | You skipped training AND deleted the `.pth` file. Either re-download it from GitHub or run `python train.py`. |
| `streamlit: command not found` | You forgot to activate the venv. Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux), then `pip install -r requirements.txt` again. |
| Training is slow | Normal on CPU (~3-8 min for 5 epochs). Close Chrome tabs. GPU/Mac MPS is picked automatically if available. |
| Blurry digits | Normal for a tiny 2-number brain trained 5 epochs. Train longer (edit `EPOCHS = 5` → `20` in `train.py`) for sharper doodles. |
| Port already in use | Another Streamlit is running. Press `Ctrl+C` in the old window, or run `streamlit run app.py --server.port 8502`. |

---

## Tech stack

Python · PyTorch + Torchvision · Streamlit · Pandas · Matplotlib — nothing else.

Model: Conv VAE, `z_dim = 2`, loss = BCE reconstruction + KL to N(0, I). ~5 epochs, Adam `lr=1e-3`, batch 128, seed 42.

---

## License

MIT — do whatever you want with it, just keep the credit. See `LICENSE`.

Made for learning. If a total beginner got it running from this README, it did its job. ⭐ Star it if it helped!
