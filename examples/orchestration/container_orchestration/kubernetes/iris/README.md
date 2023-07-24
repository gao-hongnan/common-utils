# Kubernetes

- [Kubernetes](#kubernetes)
  - [What is a port](#what-is-a-port)
  - [Terminology](#terminology)
  - [Setup Kubernetes Locally](#setup-kubernetes-locally)
  - [Local Kubernetes Cluster](#local-kubernetes-cluster)
    - [What's a Kubernetes Cluster?](#whats-a-kubernetes-cluster)
    - [What's a Pod?](#whats-a-pod)
    - [What's a Service?](#whats-a-service)
      - [More on Services IMPORTANT](#more-on-services-important)
    - [What's a Deployment?](#whats-a-deployment)
    - [Summary](#summary)
    - [Setting Up](#setting-up)
    - [Cluster is Automatically Configured](#cluster-is-automatically-configured)
  - [Useful Commands](#useful-commands)
    - [kubectl cluster-info](#kubectl-cluster-info)
    - [kubectl get nodes](#kubectl-get-nodes)
    - [kubectl describe node](#kubectl-describe-node)
    - [kubectl get pods](#kubectl-get-pods)
    - [kubectl get services](#kubectl-get-services)
    - [kubectl get deployments](#kubectl-get-deployments)
      - [Create a Deployment](#create-a-deployment)
      - [Verify the Deployment](#verify-the-deployment)

Let's think about Kubernetes as a containerized shipping company.

- **Kubernetes Cluster**: This is akin to the entire shipping company -
    including its infrastructure, offices, communication systems, etc. It's the
    entire system that makes the shipping operation possible.

- **Node**: Each node is like a shipping port in this analogy. Just as a port
    has resources like cranes, forklifts, staff, and space to accommodate
    containers, a node has resources like CPU, memory, and storage to run
    containers.

- **Pod**: A pod is analogous to an individual shipping container. Each
    shipping container can carry one or more types of goods (like a Pod can run
    one or more containers). When a container is closed and sealed, it provides
    a self-contained environment for the goods, much like a pod provides a
    self-contained environment for running containers.

- **Service**: A service can be thought of as a shipping manifest or
    schedule - it's the information that allows the transportation of containers
    to the correct location. Just like a shipping manifest might specify that
    certain containers need to be loaded onto a specific ship at a specific time
    to reach a specific destination, a Kubernetes service ensures network
    traffic is correctly routed to the right pods based on their selectors.

- **Deployment**: A deployment in Kubernetes could be compared to a shipping
    contract. It ensures a certain number of containers (pods) are running and
    if a container (pod) fails, the contract (deployment) ensures another one is
    shipped out (rescheduled) to meet the contract's requirements.

- **Kube-scheduler**: The scheduler is like the harbor master, who decides
    where each incoming container should be placed based on the available
    resources and the needs of the container.

- **Kubelet**: Kubelet is like a crane operator at each port. It communicates
    with the harbor master (scheduler) to understand what needs to be done and
    then manages the containers (pods) at its port (node), starting them,
    stopping them, and checking their health.

This analogy may not cover all aspects of Kubernetes but should help provide a
high-level understanding of how its different components interact with each
other.

## What is a port

Let's break this down with an analogy. Imagine an apartment building that is
home to many people. The apartment building address is akin to an IP address -
it tells you where to find the building in the city. Now, imagine you're a mail
delivery person with a letter to deliver to a resident of this apartment
building. Knowing the apartment building's address isn't enough; you also need
to know the specific apartment number to deliver the mail correctly. In this
analogy, the apartment number is akin to a port number.

In the realm of computer networking, an IP address helps us find the specific
computer on a network, but once we get to that computer, we need to know which
application we're trying to reach. That's where ports come in.

Here's a more technical example: Let's say you have a server running a web
service and a database service. The web service is running on port 80 (the
standard port for HTTP traffic) and the database is running on port 3306 (the
standard port for MySQL).

When a user's browser sends a request to your server's IP address, it also
specifies that it wants to connect to port 80 because it wants to communicate
with the web service. If it tried to connect to port 3306, it would be trying to
communicate with the database, which wouldn't understand the browser's request.

As such, ports are essential for directing network traffic to the correct
application on a computer. If every service was just listening on a single port,
they would all be trying to handle the same traffic, which would result in
conflicts and errors. By having services listen on different ports, they can all
coexist on a single computer without interfering with each other.

In conclusion, a port is a communication endpoint. It is associated with a
specific process or service on a computer and allows that process or service to
be accessed over a network. Ports help the operating system to route incoming
traffic to the appropriate software application.

## Terminology

- **Cluster**: A set of Nodes that run containerized applications. A cluster
    consists of a master node and a set of worker nodes.
- **Node**: A node is a worker machine in Kubernetes and may be either a
    virtual or a physical machine, depending on the cluster. Each node has the
    services necessary to run Pods and is managed by the master components. The
    services on a node include the container runtime, kubelet, and kube-proxy.
- **Pod**: The smallest and simplest unit in the Kubernetes object model that
    you create or deploy. A Pod represents processes running on your Cluster.

  - So a node can run many pods.
  - And a pod can have a collection of containers.

- **Service**: An abstract way to expose an application running on a set of
    Pods as a network service. Kubernetes gives Pods their own IP addresses and
    a single DNS name for a set of Pods, and can load-balance across them via a
    Service.
- **Deployment**: A Deployment provides declarative updates for Pods and
    ReplicaSets.
- **ReplicaSet**: A ReplicaSet ensures that a specified number of pod replicas
    are running at any given time.
- **Namespace**: Kubernetes supports multiple virtual clusters backed by the
    same physical cluster. These virtual clusters are called namespaces. They
    let you partition resources into logically named groups.

## Setup Kubernetes Locally

The command `kubectl config use-context docker-desktop` is for switching your
current context to `docker-desktop`. The context in Kubernetes is used to
determine which cluster you're interacting with, as well as which user and
namespace within that cluster.

This command assumes that you have Docker Desktop installed on your machine, and
you have enabled the Kubernetes feature that comes with Docker Desktop. This
would create a single-node Kubernetes cluster on your machine, which is great
for local testing and development.

When you run Kubernetes on Docker Desktop, it comes with a context called
`docker-desktop`. By using the `kubectl config use-context docker-desktop`
command, you're telling `kubectl` (the command line tool for interacting with
Kubernetes) to interact with the Kubernetes cluster that's running via Docker
Desktop.

```bash
kubectl config use-context docker-desktop
```

## Local Kubernetes Cluster

### What's a Kubernetes Cluster?

A Kubernetes cluster is made up of a set of nodes, which are the machines
(physical or virtual) where your workloads (containers) run.

So a Kubernetes cluster can have many nodes. A node is a worker machine and may
be either a virtual or a physical machine, depending on the cluster. Each node
contains the services necessary to run Pods and is managed by the master
components.

There are two types of nodes:

1. **Worker nodes**: These nodes run the workloads. They are controlled by the
   master node and they run the actual containers of the applications.

2. **Master node**: This node controls the worker nodes. It schedules the
   deployment of the containers to the worker nodes. The master node also keeps
   track of the state of the cluster.

A cluster with more than one node allows you to run your applications and
services in a high-availability setup, where one or more nodes can fail, and
your applications will still be available. It also enables you to scale out your
application or service by running more pods on more nodes.

### What's a Pod?

Pods are the smallest deployable units in Kubernetes that can be created and
managed. Each Pod represents a single instance of a running process in the
cluster and can contain one or more containers. Containers within a Pod share an
IP address and port space, and can communicate with one another using
`localhost`. They can also share storage volumes.

### What's a Service?

Services are an abstract way to expose an application running on a set of Pods
as a network service. Kubernetes gives Pods their own IP addresses and a single
DNS name for a set of Pods, and can load-balance across them via a Service.

#### More on Services IMPORTANT

In Kubernetes, a Service is an abstraction which defines a logical set of Pods
and a policy by which to access them. Services enable a loose coupling between
dependent Pods.

A Kubernetes Service is used to expose an application running on a set of Pods
as a network service. Think of Pods as the running processes and Services as the
front-end load balancer and traffic router.

With Kubernetes, you don’t need to modify your application to use an unfamiliar
service discovery mechanism. Kubernetes gives Pods their own IP addresses and a
single DNS name for a set of Pods, and can load-balance across them.

The set of Pods targeted by a Service is usually determined by a selector. If a
Service has a selector, the Service controller will continuously monitor the
running Pods in the Kubernetes cluster, and will create and update the Service’s
endpoint list (the set of Pods that the Service’s selector will match) whenever
the set of Pods in the cluster changes.

There are different types of Services in Kubernetes:

1. **ClusterIP**: Exposes the Service on a cluster-internal IP. This makes the
   Service only reachable from within the cluster. This is the default
   ServiceType.

2. **NodePort**: Exposes the Service on each Node’s IP at a static port (the
   NodePort). A ClusterIP Service, to which the NodePort Service routes, is
   automatically created.

3. **LoadBalancer**: Exposes the Service externally using a cloud provider’s
   load balancer. NodePort and ClusterIP Services, to which the external load
   balancer routes, are automatically created.

4. **ExternalName**: Maps the Service to the contents of the `externalName`
   field (e.g. `foo.bar.example.com`), by returning a CNAME record.

Each Service has a `type` field which defines the type of Service it is. The
type dictates how the Service is exposed to network traffic.

A Service in Kubernetes is crucial when you're dealing with dynamic Pod creation
and destruction. When a Pod dies and gets replaced, it will have a different IP,
but the Service ensures that the traffic will always get to the intended
destination.

### What's a Deployment?

A Deployment provides declarative updates for Pods and ReplicaSets. You describe
a desired state in a Deployment, and the Deployment controller changes the
actual state to the desired state at a controlled rate.

### Summary

Here's a simplified hierarchy:

- Cluster: A set of Nodes.
- Node: An individual machine that runs containers.
- Pod: The smallest and simplest unit in the Kubernetes object model that you
    create or deploy. A Pod represents processes running on your Cluster.
- Service: An abstract way to expose an application running on a set of Pods
    as a network service.

It's important to note that while Pods run on nodes, Services operate at the
cluster level and are not tied to specific nodes. Instead, a Service routes
traffic to appropriate Pods, no matter which node they are running on.

### Setting Up

You can set up a local Kubernetes cluster using Minikube or Docker Desktop. For
simplicity, we will use Docker Desktop's Kubernetes feature.

Before proceeding, please make sure you have Docker Desktop installed and that
the Kubernetes feature is enabled (Settings -> Kubernetes -> Enable Kubernetes).
It might take a few minutes for the Kubernetes cluster to start up.

Now that we have our Kubernetes cluster running, we will interact with it using
`kubectl`, the Kubernetes command-line tool.

You can verify that your cluster is up and running with the following command:

```bash
kubectl cluster-info
```

You should see something like this:

```bash
Kubernetes control plane is running at https://kubernetes.docker.internal:6443
CoreDNS is running at https://kubernetes.docker.internal:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
```

The output from `kubectl cluster-info` command gives you a high-level overview
of your Kubernetes cluster's master node (control plane) and some key services
running within the cluster.

1. **Kubernetes control plane is running at
   <https://kubernetes.docker.internal:6443>**: This line tells you where the
   Kubernetes master, or the control plane, is running. The control plane is the
   collection of components that control and manage the state of the cluster,
   including scheduling and orchestrating the containerized applications,
   maintaining the desired state of the applications, scaling applications based
   on demand, and rolling out new updates. The API server, which is part of the
   control plane, is exposed at the given URL
   (<https://kubernetes.docker.internal:6443>). You can interact with the API
   server (and thereby the control plane) using `kubectl` commands, Kubernetes
   Dashboard, REST calls, etc.

2. **CoreDNS is running at
   <https://kubernetes.docker.internal:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy>**:
   This line provides the endpoint where the CoreDNS service is running in your
   cluster. CoreDNS is a flexible, extensible DNS server that can serve as the
   Kubernetes cluster DNS. In a Kubernetes cluster, CoreDNS is used to discover
   services within the same cluster (this is known as service discovery). The
   URL shows that the CoreDNS service is running in the `kube-system` namespace
   (a special namespace for system components managed by Kubernetes itself), and
   it can be accessed via the Kubernetes API.

These details are primarily of interest when you're debugging cluster issues or
when you need to directly interact with these components. Under normal
circumstances, you won't need to use these URLs directly, as you'd interact with
the cluster primarily through `kubectl` or other Kubernetes tools.

### Cluster is Automatically Configured

When you enable Kubernetes in Docker Desktop, it automatically creates a local
cluster for you, so you don't need to manually create a cluster. This local
cluster is a single-node cluster, meaning it only contains one machine (your
own).

## Useful Commands

Before we even deploy any applications to our cluster, let's take a look at some
basic `kubectl` commands that you'll use frequently.

### kubectl cluster-info

The `kubectl cluster-info` command gives you a high-level overview of your
Kubernetes cluster's master node (control plane) and some key services running
within the cluster.

### kubectl get nodes

The `kubectl get nodes` command lists all the nodes that are part of the
cluster. In our case, we only have one node, which is our local machine.

```bash
kubectl get nodes
```

You should see something like this:

```bash
NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   27m   v1.25.0
```

which shows that we have one **node** called `docker-desktop` that is ready to
accept workloads.

### kubectl describe node

The `kubectl describe node` command gives you detailed information about a
particular node in the cluster. You can use this command to get information
about the node's capacity, the pods running on the node, and the node's
allocatable resources.

```bash
kubectl describe node <NODE-NAME>
```

where `<NODE-NAME>` is the name of the node you want to get information about.
In this case, it is `docker-desktop`.

```bash
kubectl describe node docker-desktop
```

You should see something like this:

```bash
Name:               docker-desktop
Roles:              control-plane
Labels:             beta.kubernetes.io/arch=arm64
                    beta.kubernetes.io/os=linux
                    kubernetes.io/arch=arm64
                    kubernetes.io/hostname=docker-desktop
                    kubernetes.io/os=linux
                    node-role.kubernetes.io/control-plane=
                    node.kubernetes.io/exclude-from-external-load-balancers=
Annotations:        kubeadm.alpha.kubernetes.io/cri-socket: unix:///var/run/cri-dockerd.sock
                    node.alpha.kubernetes.io/ttl: 0
                    volumes.kubernetes.io/controller-managed-attach-detach: true
CreationTimestamp:  Sat, 22 Jul 2023 20:13:59 +0800
Taints:             <none>
Unschedulable:      false
Lease:
  HolderIdentity:  docker-desktop
  AcquireTime:     <unset>
  RenewTime:       Sat, 22 Jul 2023 20:42:24 +0800
Conditions:
  Type             Status  LastHeartbeatTime                 LastTransitionTime                Reason                       Message
  ----             ------  -----------------                 ------------------                ------                       -------
  MemoryPressure   False   Sat, 22 Jul 2023 20:42:06 +0800   Sat, 22 Jul 2023 20:13:58 +0800   KubeletHasSufficientMemory   kubelet has sufficient memory available
  DiskPressure     False   Sat, 22 Jul 2023 20:42:06 +0800   Sat, 22 Jul 2023 20:13:58 +0800   KubeletHasNoDiskPressure     kubelet has no disk pressure
  PIDPressure      False   Sat, 22 Jul 2023 20:42:06 +0800   Sat, 22 Jul 2023 20:13:58 +0800   KubeletHasSufficientPID      kubelet has sufficient PID available
  Ready            True    Sat, 22 Jul 2023 20:42:06 +0800   Sat, 22 Jul 2023 20:14:30 +0800   KubeletReady                 kubelet is posting ready status
Addresses:
  InternalIP:  192.168.65.4
  Hostname:    docker-desktop
Capacity:
  cpu:                4
  ephemeral-storage:  61202244Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  hugepages-32Mi:     0
  hugepages-64Ki:     0
  memory:             8039920Ki
  pods:               110
Allocatable:
  cpu:                4
  ephemeral-storage:  56403987978
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  hugepages-32Mi:     0
  hugepages-64Ki:     0
  memory:             7937520Ki
  pods:               110
System Info:
  Machine ID:                 3bf085c6-a864-406e-93d6-97a454c9ef4d
  System UUID:                3bf085c6-a864-406e-93d6-97a454c9ef4d
  Boot ID:                    6f896961-3d86-48e5-827d-925165bbd11b
  Kernel Version:             5.10.124-linuxkit
  OS Image:                   Docker Desktop
  Operating System:           linux
  Architecture:               arm64
  Container Runtime Version:  docker://20.10.17
  Kubelet Version:            v1.25.0
  Kube-Proxy Version:         v1.25.0
Non-terminated Pods:          (9 in total)
  Namespace                   Name                                      CPU Requests  CPU Limits  Memory Requests  Memory Limits  Age
  ---------                   ----                                      ------------  ----------  ---------------  -------------  ---
  kube-system                 coredns-95db45d46-mqmxg                   100m (2%)     0 (0%)      70Mi (0%)        170Mi (2%)     28m
  kube-system                 coredns-95db45d46-wmt77                   100m (2%)     0 (0%)      70Mi (0%)        170Mi (2%)     28m
  kube-system                 etcd-docker-desktop                       100m (2%)     0 (0%)      100Mi (1%)       0 (0%)         28m
  kube-system                 kube-apiserver-docker-desktop             250m (6%)     0 (0%)      0 (0%)           0 (0%)         28m
  kube-system                 kube-controller-manager-docker-desktop    200m (5%)     0 (0%)      0 (0%)           0 (0%)         28m
  kube-system                 kube-proxy-tg8rn                          0 (0%)        0 (0%)      0 (0%)           0 (0%)         28m
  kube-system                 kube-scheduler-docker-desktop             100m (2%)     0 (0%)      0 (0%)           0 (0%)         28m
  kube-system                 storage-provisioner                       0 (0%)        0 (0%)      0 (0%)           0 (0%)         27m
  kube-system                 vpnkit-controller                         0 (0%)        0 (0%)      0 (0%)           0 (0%)         27m
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests    Limits
  --------           --------    ------
  cpu                850m (21%)  0 (0%)
  memory             240Mi (3%)  340Mi (4%)
  ephemeral-storage  0 (0%)      0 (0%)
  hugepages-1Gi      0 (0%)      0 (0%)
  hugepages-2Mi      0 (0%)      0 (0%)
  hugepages-32Mi     0 (0%)      0 (0%)
  hugepages-64Ki     0 (0%)      0 (0%)
Events:
  Type    Reason                   Age                From             Message
  ----    ------                   ----               ----             -------
  Normal  Starting                 28m                kube-proxy
  Normal  Starting                 28m                kubelet          Starting kubelet.
  Normal  NodeHasSufficientMemory  28m (x8 over 28m)  kubelet          Node docker-desktop status is now: NodeHasSufficientMemory
  Normal  NodeHasNoDiskPressure    28m (x8 over 28m)  kubelet          Node docker-desktop status is now: NodeHasNoDiskPressure
  Normal  NodeHasSufficientPID     28m (x7 over 28m)  kubelet          Node docker-desktop status is now: NodeHasSufficientPID
  Normal  NodeAllocatableEnforced  28m                kubelet          Updated Node Allocatable limit across pods
  Normal  RegisteredNode           28m                node-controller  Node docker-desktop event: Registered Node docker-desktop in Controller
```

Here are some key aspects of the information provided:

1. **Node Information**: The basic information like node name
   ("docker-desktop"), the roles it has in the cluster ("control-plane"),
   labels, annotations, taints, and conditions related to the node's status are
   shown.

2. **System Information**: Detailed system info like the Machine ID, System
   UUID, Boot ID, Kernel Version, OS Image, Operating System, Architecture,
   Container Runtime Version, Kubelet Version, and Kube-Proxy Version is given.

3. **Resources**: It shows the capacity and the allocatable resources for the
   node like CPU, memory, ephemeral-storage, and pods.

4. **Pods**: It shows the pods running on this node, along with their CPU and
   memory usage, and their age (how long they've been running).

5. **Allocated Resources**: It displays the total resources requested by all
   pods on the node and their limits. In the given output, CPU requests are at
   850m (21%) of total capacity and Memory requests are at 240Mi (3%) of total
   capacity.

6. **Events**: It shows any events related to this node. The events can be very
   helpful in troubleshooting any issues with the node.

In this case, the node is a Docker Desktop node, and it is a control-plane
(meaning it runs Kubernetes master components). It's operating system is Linux
and its architecture is arm64. It has 4 CPUs, about 8GB of memory, and can
support 110 pods. Currently, it's running system pods like coredns, etcd,
kube-apiserver, kube-controller-manager, kube-proxy, kube-scheduler, a
storage-provisioner, and a vpnkit-controller.

This is generally the information you would need when troubleshooting issues
related to specific nodes, their workloads, or capacity planning.

### kubectl get pods

A Pod is the smallest and simplest unit in the Kubernetes object model that you
create or deploy. A Pod represents processes running on your cluster. Pods can
contain one or more containers. They are typically used for running your
applications.

When you create a Deployment, it creates a Pod to host the application instance.
The command `kubectl get pods` shows you all the pods that are running in your
cluster (or in your namespace, if you have set one).

Here's how you can use it:

```bash
kubectl get pods
```

or equivalently:

```bash
kubectl get pods --namespace=default
```

because `default` is the default namespace.

You should see something like this:

```bash
No resources found in default namespace.
```

This means that there are no pods running in your default namespace.

There are however pods in the namespace `kube-system`:

```bash
kubectl get pods --all-namespaces
```

You should see something like this:

```bash
NAMESPACE     NAME                                     READY   STATUS    RESTARTS        AGE
kube-system   coredns-95db45d46-mqmxg                  1/1     Running   0               57m
kube-system   coredns-95db45d46-wmt77                  1/1     Running   0               57m
kube-system   etcd-docker-desktop                      1/1     Running   0               57m
kube-system   kube-apiserver-docker-desktop            1/1     Running   0               57m
kube-system   kube-controller-manager-docker-desktop   1/1     Running   0               57m
kube-system   kube-proxy-tg8rn                         1/1     Running   0               57m
kube-system   kube-scheduler-docker-desktop            1/1     Running   0               57m
kube-system   storage-provisioner                      1/1     Running   0               56m
kube-system   vpnkit-controller                        1/1     Running   4 (4m49s ago)   56m
```

### kubectl get services

Similarly, if we check the default namespace:

```bash
kubectl get services --namespace=default
```

We will see

```bash
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   96m
```

which is the Kubernetes API server.

So, the Kubernetes service named "kubernetes" in the default namespace is a
special service created automatically by Kubernetes. It provides a way for other
components to access the API server. The API server is the central management
entity of a Kubernetes cluster and provides a RESTful API interface for managing
the components of the cluster.

Let's break down the output you got:

- **NAME**: The name of the service is "kubernetes".

- **TYPE**: "ClusterIP" means that this service is reachable within the
    cluster. Other types can include NodePort and LoadBalancer.

- **CLUSTER-IP**: This is the IP address that other components within the
    cluster can use to communicate with the service. In this case, "10.96.0.1"
    is the internal IP address for the Kubernetes API server within the cluster.

- **EXTERNAL-IP**: This field would show an external IP address if one were
    assigned, but the `<none>` here indicates that the API server is not
    intended to be reached from outside of the cluster (which is typically the
    case, for security reasons).

- **PORT(S)**: This indicates that the API server is listening on TCP port
    443, which is the standard port for HTTPS connections. The communication
    with the API server is encrypted and secure.

- **AGE**: This shows how long the service has been running. Here, it's been
    running for 96 minutes.

So, in short, this service is the internal representation of your Kubernetes API
server within the cluster, and provides a way for other components in the
cluster to interact with it.

Similarly, we can also see the services in the `kube-system` namespace:

```bash
kubectl get services --namespace=kube-system
```

You should see something like this:

```bash
NAMESPACE     NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)                  AGE
kube-system   kube-dns     ClusterIP   10.96.0.10   <none>        53/UDP,53/TCP,9153/TCP   98m
```

### kubectl get deployments

Let's create a simple nginx deployment to see how this command works.

#### Create a Deployment

```bash
kubectl create deployment nginx --image=nginx
```

Nginx is a popular open-source web server software. It can also be used as a
reverse proxy, load balancer, and HTTP cache. It's widely used because of its
high performance and stability.

In the context of the command `kubectl create deployment nginx --image=nginx`,
we're creating a new Deployment in Kubernetes. The Deployment is named "nginx"
and it's going to run a container with the Nginx software.

This container will be created from the `nginx` Docker image, which is a
lightweight, stand-alone, executable package that includes everything needed to
run a piece of software, including the code, a runtime, libraries, environment
variables, and config files. The Nginx Docker image is available on Docker Hub,
which is a public registry for Docker images.

So, to simplify, you're telling Kubernetes: "Hey, create a new deployment using
the Nginx Docker image and keep it running for me." Kubernetes will then ensure
that the Nginx server is always running inside a Pod.

This is just a test Deployment, in reality, you would probably be deploying your
own custom applications to Kubernetes. The Nginx server is just being used for
illustrative purposes.

#### Verify the Deployment

```bash
kubectl get deployments
```

---

This command will list all the pods in your default namespace.

If you created the nginx deployment as described in the previous message, you
should see a pod named similar to `nginx-xxxxxxxxxx-xxxxx`. This Pod is hosted
on a Node in your cluster.

You can describe a specific pod in more detail with the following command:

```bash
kubectl describe pod <pod-name>
```

This will give you detailed information about the specific pod, including the
state of the pod, recent events, and other useful debugging information.

So while `kubectl get pods` might not be the command you use the most,
understanding pods and how they work is critical to understanding Kubernetes as
a whole.

---

In the next step, you'd typically start deploying your applications to the
cluster (such as the Docker containers you've mentioned previously) and begin
testing or development.

I would be glad to help you get a good understanding of Kubernetes (K8s) within
the next five days. Here's an outline of the topics we're going to cover:

- Introduction to Kubernetes Concepts
- Kubernetes Manifest Files and Object Types
- Setting Up a Local Kubernetes Cluster
- Deploying Applications using Docker Containers
- ConfigMaps and Secrets
- Scaling and Rolling Updates
- Services and Networking
- Persistent Storage and Volumes
- Health Checks and Monitoring

Let's start with the **Introduction to Kubernetes Concepts**:

1. **Pods**: The smallest deployable units of computing that you can create and
   manage in Kubernetes. A Pod represents a single instance of a running process
   in a cluster and can contain one or more containers.

2. **Services**: An abstraction that defines a logical set of Pods and a policy
   by which to access them. Services enable loose coupling between dependent
   Pods.

3. **Volumes**: A directory, possibly with some data in it, which is accessible
   to the Containers in a Pod. Kubernetes supports many types of Volumes,
   allowing you to work with files and storage systems, including your local
   filesystem and network storage systems like NFS.

4. **Namespaces**: Kubernetes supports multiple virtual clusters backed by the
   same physical cluster. These virtual clusters are called namespaces. They let
   you partition resources into logically named groups.

5. **Deployments**: An API object that manages a replicated application,
   typically by running Pods with the same container(s).

Now, let's get into **Kubernetes Manifest Files and Object Types**:

A Kubernetes Manifest File is a definition in YAML or JSON format that defines
an object or a set of objects in Kubernetes. It's usually used to create,
modify, or control Kubernetes resources like Pods, Deployments, Services, etc.

For example, let's take a simple YAML definition for a Kubernetes Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
    name: iris-training-deployment
    labels:
        app: iris-training
spec:
    replicas: 1
    selector:
        matchLabels:
            app: iris-training
    template:
        metadata:
            labels:
                app: iris-training
        spec:
            containers:
                - name: iris-training
                  image: iris-training:v1
                  ports:
                      - containerPort: 8501
```

This manifest file describes a Deployment which runs 3 replicas of the
`my-app:1.0.0` container. Each replica is exposing port 8080.

The YAML you've quoted is a generic example that demonstrates the typical
structure of a Kubernetes deployment manifest. To adapt it for your specific
Docker images, you'd need to replace my-app:1.0.0 with the tags of your actual
images (`iris-training:v1` and `iris-app:v1`), and ensure that the
`containerPort` values match the ports your apps are actually using.

Here's how you might adapt the deployment for your training Docker image:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
    name: iris-training-deployment
    labels:
        app: iris-training
spec:
    replicas: 1
    selector:
        matchLabels:
            app: iris-training
    template:
        metadata:
            labels:
                app: iris-training
        spec:
            containers:
                - name: iris-training
                  image: iris-training:v1
                  ports:
                      - containerPort: 8501
```

For the serving Docker image:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
    name: iris-app-deployment
    labels:
        app: iris-app
spec:
    replicas: 1
    selector:
        matchLabels:
            app: iris-app
    template:
        metadata:
            labels:
                app: iris-app
        spec:
            containers:
                - name: iris-app
                  image: iris-app:v1
                  ports:
                      - containerPort: 8501
```

Note that these YAML files only define the deployments and don't address issues
like persistent storage or inter-container communication. Depending on your
apps' needs, you may need to include additional Kubernetes objects in your
setup. I assumed that your application runs on port 8501 as per the Docker
command in your previous question. If it is different, please modify the
containerPort accordingly.

In the next session, we will learn how to **Set Up a Local Kubernetes Cluster**.
