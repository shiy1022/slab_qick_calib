
nexp = 100
qubit_list =[0,1,2,3,4,5]
expts = 200
t1_data = [[] for _ in qubit_list]

for i in range(nexp):
    for j, qi in enumerate(qubit_list):
        t1_cont = meas.T1StarkPowerSingle(cfg_dict, qi=qi, params={'active_reset':True, "soft_avgs":4})
        t1_data[j].append(t1_cont['t1'])

plt.figure()
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
ax = ax.flatten()
for i, qi in enumerate(qubit_list):
    ax[i].plot(t1_data[i], 'o')
    ax[i].set_title(f'Q{qi}')
    ax[0].pcolormesh(t1_data[i])




        