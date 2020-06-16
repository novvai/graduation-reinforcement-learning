def loginto(path, msg):
    f = open(path, "a")
    f.write(msg)
    f.close()