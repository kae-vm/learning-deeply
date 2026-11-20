# one time setup:
ssh into the remote PC, run `pip install notebook`
follow the `make_kernel` announcement steps on the idl brightspace

# for each time:
ssh into the remote PC, and cd into the learning-deeply folder.
run `python3 -m notebook --no-browser --port=8888`
it'll provide a link in the format of "localhost:8888/tree?token=.....", copy that

on your own pc:
ensure your jupyter is closed, and nothing is running there
run `ssh -N -L 8888:localhost:8888 -X username@dm009a.student.liacs.nl -J username@ssh.liacs.nl`, and enter the passwords
paste the link into your browser, and change the kernel to Python (idl2026)
run! it might take a few to set up, but it works as intended after
