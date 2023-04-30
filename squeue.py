

OUR_USERS = ['ncavagne']

class Table:
    def __init__(self, header=None):
        self.header = header; self.rows = []
    def add(self, row):
        if self.header != None or len(self.rows) != 0:
            assert len(row) == self.num_cols(), f"Different len: {len(row)} {self.num_cols()}"
        self.rows.append(row)
    def num_cols(self):
        if self.header != None: return len(self.header)
        else: return len(self.rows[0])
    def num_rows(self):
        return len(self.rows)
    def show(self, line_number=True, sep="   "):
        def _s_(obj): return str(obj).replace("\n", "")
        if self.header == None and len(self.rows) == 0: return "Tabella vuota"
        if self.header != None: all_rows = [self.header] + self.rows
        else: all_rows = self.rows
        col_widths = [max([len(_s_(all_rows[r][c])) for r in range(len(all_rows))]) for c in range(self.num_cols())]
        table = [[f"{_s_(all_rows[r][c]):<{col_widths[c]}}" for c in range(self.num_cols())] for r in range(len(all_rows))]
        if line_number: table = "\n".join([f"{i:2d}  " + sep.join(row) for i, row in enumerate(table)])
        else: table = "\n".join([sep.join(row) for row in table])
        return table
    def __repr__(self): print(self.show()); return ""


import os
# Read jobs info from squeue
jobs = os.popen("squeue -o '%8A %12u %Z %75j %10B %5C %7m %10p %12r %15b %.2t %14l %12M  %20V %5D'").read().split("\n")

def format_date(date): return date[8:10] + " " + date[11:16]

# Some formatting of the jobs
jobs = [jobs[0], *sorted(jobs[1:-1])]
jobs = [[item for item in job.split(" ") if item != ""] for job in jobs]
header = jobs[0]
header = [*header[:4], header[4][5:], header[5], header[6][4:], *header[7:9], "GPUS", *header[10:12], "RUN_TIME", *header[13:]]
jobs = [[*j[:2], j[2].split("/")[-1], *j[3:7], j[7][5:], *j[8:13], format_date(j[13]), j[14]] for j in jobs[1:]]

# Filter away jobs from other users, and put our jobs into the Table object, use to pretty-print the jobs info
our_jobs = [j for j in jobs if j[1] in OUR_USERS]
table = Table(header)
for j in our_jobs: table.add(j)

print(table.show(sep="  "))

# Filter jobs waiting in queue
running_jobs = [j for j in jobs if j[8] == "None"]
nodes = [int(j[14]) for j in running_jobs]
cpus = [int(j[5]) for j in running_jobs]  # CPUs
rams = [j[6] for j in running_jobs]
rams = [(int(float(r.replace("M", "")))//1000 if "M" in r else int(float(r.replace("G", ""))))*n for r, n in zip(rams, nodes)]  # RAM
gpus = [j[9].replace("requir", "").replace("gres:", "").replace("gpu:", "").strip() for j in running_jobs]
gpus = [int(g)*n if g.isdigit() else 0 for g, n in zip(gpus, nodes)]  # GPUs
assert len(nodes) == len(cpus) == len(rams) == len(gpus)
print(f"Being used {sum(gpus)} / {980*4} GPUs and {sum(cpus)} / {980*128} CPUs and {sum(rams)//1000} / {980*240//1000} TB of RAM")

nodes_in_use = os.popen("sinfo -o%A").read().split("\n")[1].split("/")
print(f"Allocated nodes: {nodes_in_use[0]}, idle nodes: {nodes_in_use[1]} (PS run '$ sinfo -o%A' to get this info)")

for count, job in enumerate([j for j in jobs if j[8] == "None"]):
    if OUR_USERS[0] in job:
        break

print(f"There are {count} jobs before your first (if you're not waiting in queue then ignore this)")

