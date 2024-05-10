# Load kge log
with open('kge.log', 'r') as f:
    lines = f.read().split('\n')
epoch_times_lines = [i for i, s in enumerate(lines) if 'epoch_time' in s]

# Parse runtimes
epoch_times = {}
x = 0
for i in epoch_times_lines:
    if 'backward_time' in lines[i-4]:  # Check we are on a training pass
        epoch_num = int(lines[i-1].split(': ')[-1])
        time = float(lines[i].split(': ')[-1])
        if epoch_num not in epoch_times:
            epoch_times[epoch_num] = time
        else:
            try:
                assert epoch_times[epoch_num] == time
            except AssertionError as e:
                print(f'ERROR: Found different forward times for epoch {epoch_num}\n')
                raise e

# Write to csv
with open('epoch_runtimes.csv', 'w') as f:
    f.write('epoch,runtime\n')
    for epoch in epoch_times:
        f.write(f'{epoch},{epoch_times[epoch]}\n') 