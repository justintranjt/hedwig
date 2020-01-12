import os, re, sys

import matplotlib


havedisp = "DISPLAY" in os.environ
if havedisp:
    import matplotlib.pyplot as plt
else:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 4:
        print("Usage: python devacc.py <Dataset Title> <inputfile>.txt {DevAcc,DevPr,DevRecall,DevF1,DevLoss,...} {...}")
        exit()

    title = sys.argv[1]

    files = []
    for x in sys.argv[2:]:
        if ".txt" in x:
            files.append(x)


    def dropout(fn):
        return float(re.search("embed-droprate-(\d\.\d)", fn).group(1))

    dropouts = [dropout(fn) for fn in files]
    dataset = files[0][:4]
    date = (files[0][-17:])[:6]

    d = "(?P<%s>\d+)"
    e = "(?P<%s>\d+\.\d+)"
    s = "(?P<%s>\d+/\d+)"
    w = "\s+"

    pcs = [
        ["Time", d],
        ["Epoch", d],
        ["Iteration", d],
        ["Progress", s],
        ["Dev/Acc.", e],
        ["Dev/Pr.", e],
        ["Dev/Recall", e],
        ["Dev/F1", e],
        ["Dev/Loss", e],
    ]

    re1 = w.join([p[0] for p in pcs])
    #print(re1)

    for p in pcs:
        p[0] = p[0].replace("/","").replace(".", "")

    keys = [p[0] for p in pcs]

    print("Plotting files:", files)
    print("Testing dropouts:", dropouts)
    data = {y: {x[0]: [] for x in pcs} for y in dropouts}
    print(data)
    re2 = w.join([p[1] % p[0] for p in pcs])
    #print(re2)

    #     "Time Epoch Iteration Progress     Dev/Acc. Dev/Pr.  Dev/Recall   Dev/F1       Dev/Loss"
    #      14369    20      7300    20/30     0.7401   0.9446   0.7275       0.8219       1.6298

    for fn in files:
        y = dropout(fn)
        with open(fn, "r") as f:
            prev = None
            for ln in f:

                m1 = re.search(re1, ln)

                if prev:
                    m2 = re.search(re2, ln)

                    for k in keys:
                        raw = m2.group(k)
                        if k in ["Time", "Epoch", "Iteration"]:
                            x = int(raw)
                        elif k in ["Progress"]:
                            x = raw
                        else:
                            x = float(raw)
                            if "Dev" in k and "Loss" not in k:
                                x *= 100

                        data[y][k].append(x)

                prev = m1




    def disp(s):
        for y in dropouts:

            print(f"=== {s} data, dropout={y} ===")
            for x in data[y][s]:
                print(x)
            print("\n")

    def plot(s):

        fig = plt.figure()
        plt.title(f"{title} {s} over time")

        for y in dropouts:
            plt.plot(data[y]["Epoch"], data[y][s], label=f"Dropout = {y}")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(s)

        if havedisp:
            plt.show()
        fig.savefig(f"plots/{title}-{dataset}-{date}-{s}-dropouts-{dropouts[0]}-{dropouts[-1]}.png", dpi=600)

    for s in sys.argv[len(dropouts)+2:]:
        disp(s)
        plot(s)



if __name__ == "__main__":
    main()
