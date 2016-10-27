files=["predict_91","predict_92","predict_93","predict_94"]
outfile="predict_9.csv"

print("mergeing...")
out=open(outfile,"wt")
out.write("is_buy_in2month,phone_number\n")
for fi in files:
    with open(fi,"rt") as f:
        lines=f.readlines()
    out.write("".join(lines[1:]))
    out.write("\n")
    print "%s is on"%fi
print("all done.")


