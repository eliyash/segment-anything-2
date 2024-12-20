import boto3

# Replace with your instance ID
instance_id1 = 'i-06f7291958a9ee0a5'
instance_id2 = 'i-0bfcc7ce54a6f2ac2'


session = boto3.Session(profile_name='chimp')

# Create an EC2 client using the session
ec2 = session.client('ec2', region_name='us-east-1')

# ec2.reboot_instances(InstanceIds=[instance_id2]); exit()

def start_instance(instance_id):
    try:
        response = ec2.start_instances(InstanceIds=[instance_id])
        print(f'Instance {instance_id} started successfully.')
    except Exception as e:
        print(f'Error starting instance: {e}')

def get_ip(instance_id):
    response = ec2.start_instances(InstanceIds=[instance_id])

    # Get the instance ID from the response
    instance_id = response['StartingInstances'][0]['InstanceId']
    # Wait for the instance to be running
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])

    # Get the instance details
    instance = ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]

    # Get the public IP address
    public_ip = instance['PublicIpAddress']

    # Print the EC2 hostname
    print(f'ssh -i "eliahu.pem" ubuntu@ec2-{public_ip.replace(".", "-")}.compute-1.amazonaws.com')


# start_instance(instance_id1)
get_ip(instance_id2)