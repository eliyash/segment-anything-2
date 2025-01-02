import boto3

working_clearml_instance_id = 'i-0bfcc7ce54a6f2ac2'
other_instance_id = 'i-06f7291958a9ee0a5'


session = boto3.Session(profile_name='chimp')

# Create an EC2 client using the session
ec2 = session.client('ec2', region_name='us-east-1')


def reboot_instance(instance_id):
    ec2.reboot_instances(InstanceIds=[instance_id])


def start_instance(instance_id):
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
    # copy str to clipboard
    connection_command = f'ssh -i "eliahu.pem" -o StrictHostKeyChecking=no ubuntu@ec2-{public_ip.replace(".", "-")}.compute-1.amazonaws.com'
    print(connection_command)
    import pyperclip
    pyperclip.copy(connection_command)


# reboot_instance(working_clearml_instance_id)
start_instance(working_clearml_instance_id)
