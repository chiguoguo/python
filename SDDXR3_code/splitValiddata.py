inFile="feature_sd.csv"

with open(inFile) as f:
    lines=f.readlines()
header=lines[0]

for i in range(10):
    with open(inFile+str(i),"wt") as f:
        f.write(header)
        f.write("".join(lines[i*1000+1:(i+1)*1000+1]))
    print("%s wrote"%i)
print("all splited")
