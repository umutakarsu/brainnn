# Extracting Loop into its own repo

This directory is the complete, standalone `Loop` product. It currently lives at
`loop/` on the `claude/product-pivot-implementation-5ldhcn` branch of `brainnn`
only because this build session could not create a new GitHub repo (see
`DECISIONS.md` D6). Nothing here depends on `brainnn`.

## Option A — you create the repo, then push (recommended)

```bash
# 1. Create an empty repo umutakarsu/loop on GitHub (no README/license/gitignore).

# 2. From a checkout of the brainnn feature branch, split loop/ into its own
#    history-preserving branch:
git subtree split --prefix=loop -b loop-only

# 3. Push that branch as main of the new repo:
git push https://github.com/umutakarsu/loop loop-only:main
```

## Option B — copy the directory

```bash
cp -r loop /path/to/loop-repo && cd /path/to/loop-repo
git init && git add -A && git commit -m "Import Loop"
git remote add origin https://github.com/umutakarsu/loop
git push -u origin main
```

## Deploy to HuggingFace Spaces (once extracted or in place)

`README.md`'s front-matter is already a valid HF Spaces config.

```bash
huggingface-cli login                 # as umuutakarsu
huggingface-cli repo create loop --type space --space_sdk streamlit
git remote add space https://huggingface.co/spaces/umuutakarsu/loop
git push space main
```
