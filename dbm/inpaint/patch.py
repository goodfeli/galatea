import sys

f = open(sys.argv[1], 'r')
content = f.read()
f.close()

content = content.replace("/RQexec/goodfell/galatea/dbm/inpaint/mnist_zca.pkl",
                          "/u/goodfeli/galatea/dbm/inpaint/fooooooooooooo.pkl")

f = open(sys.argv[1], "w")
f.write(content)
f.close()
