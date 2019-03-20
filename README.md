The `src` folder contains a file `mkProjections.py` which shows how to use the `tensornets` library to extract the projected data points at each layer.
Right now, it tries to store all the data on disk (in the `projections` folder), 
but this is too much data.

So, you need to:

- [ ] Modify the code so that it writes only a single layer to disk
- [ ] Write another python script that takes that layer as input and outputs a bar code/persistence diagram
- [ ] Write a shell script that automates this process by calling `mkProjections.py`, then calling the other script, then deletes the intermediate results, then repeats the loop 

There's about 10T free in the `/data` partition right now,
but try to keep your usage down in the 1T range as things might start breaking if the disk gets full.
