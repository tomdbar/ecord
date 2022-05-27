#!/bin/bash
dldir="gset"
rawdir="${dldir}/_raw"

### Check for dir, if not found create it using mkdir ###
[ ! -d "$dldir" ] && mkdir -p "$dldir"
[ ! -d "$rawdir" ] && mkdir -p "$rawdir"

for i in { 1 2 3 4 5 22 23 24 25 26 43  44 45 46 47 55 60 70 }

do
  url="https://suitesparse-collection-website.herokuapp.com/MM/Gset/G${i}.tar.gz"
  file="${url##*/}"
  echo Downloading "$url" and unpacking to "${dldir}/G${i}.mtx".
  wget --no-check-certificate --no-proxy --progress=bar --show-progress -qc "$url" -O "${rawdir}/${file}"
  tar -C "${rawdir}" -xzf "${rawdir}/${file}"
  mv "${rawdir}/G${i}/G${i}.mtx" "${dldir}/G${i}.mtx"
  rm -r "${rawdir}/G${i}"
done

echo done!
