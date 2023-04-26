# server
python server.py --minfit=10 --minavl=10 --TotalDevNum=10 --num_rounds=100 --dataset='MNIST' --defense='' --method='non-iid' --train_lr=0.01 &
sleep 5

# clients
i=1
while((i<=10))
do
    python client.py --dataset='MNIST' --method='non-iid' --train_lr=0.01 --TotalDevNum=10 --DevNum=$i --batch_size=64 --defense='' --num_sen=1 --per_adv=1 &
    echo "$i is running"
    sleep 5
    let "i+=1"
done
echo "all workers start"
