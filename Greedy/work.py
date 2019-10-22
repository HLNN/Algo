def work(job, worker, workTime):
    job.sort(reverse=True)
    for i in range(len(job)):
        index = workTime.index(min(workTime))
        worker[index].append(job[i])
        workTime[index] += job[i]


if __name__ == '__main__':
    workerNum = 3
    job = [2, 14, 4, 16, 6, 5, 3]

    worker = [[] for _ in range(workerNum)]
    workTime = [0 for _ in range(workerNum)]
    work(job, worker, workTime)
    print(worker)
