# Kubernetes

- [Kubernetes](#kubernetes)
  - [The Analogies](#the-analogies)
  - [Kubernetes Cluster](#kubernetes-cluster)
    - [Definition](#definition)
    - [What is a Kubelet?](#what-is-a-kubelet)
    - [What is the difference between a master node and a worker node?](#what-is-the-difference-between-a-master-node-and-a-worker-node)
    - [How does the master node communicate with worker nodes?](#how-does-the-master-node-communicate-with-worker-nodes)
    - [How does Kubernetes handle node failures to ensure high availability?](#how-does-kubernetes-handle-node-failures-to-ensure-high-availability)
  - [Kubernetes Node](#kubernetes-node)
    - [Definition](#definition-1)
    - [Node Status](#node-status)
    - [Node Lifecycle](#node-lifecycle)
    - [Pods](#pods)
    - [Definition](#definition-2)
    - [What is a Pod?](#what-is-a-pod)
    - [Common Interview Questions on Nodes, Pods, and Containers in Kubernetes](#common-interview-questions-on-nodes-pods-and-containers-in-kubernetes)
    - [What's a Pod?](#whats-a-pod)
    - [What's a Service?](#whats-a-service)
      - [More on Services IMPORTANT](#more-on-services-important)
    - [What's a Deployment?](#whats-a-deployment)
    - [Summary](#summary)

## The Analogies

Let's try to use an analogy of a business park with various office buildings,
offices, and teams.

1. **Cluster**: Think of Kubernetes Cluster as the **management and
   organizational structure** in a large **business park**. This management
   structure ensures that the entire business park runs smoothly. They oversee
   the running of all office buildings, offices, and teams within it, making
   sure resources are allocated efficiently, new teams are housed in suitable
   offices, and any issues are dealt with swiftly.

    - Mapletree Business City is a business park in Singapore.

2. **Nodes**: These are the **office buildings** in the business park (cluster).
   Each office building (node) has a number of offices (pods) in it. The
   management (Kubernetes cluster) decides which office building (node) is best
   for each office (pod) based on the resources available.

    - Google office building in the Mapletree Business City is a node.
    - Oracle office building in the Mapletree Business City is another node.

3. **Pods**: Pods are like **individual offices** within an office building
   (node). Each office can house one or more employees (containers). Employees
   within the same office work closely together, share resources, and
   communicate easily. Similarly, pods can run one or more closely related
   containers, which share resources and a network namespace.

    Pods are the smallest deployable units of computing in a Kubernetes cluster,
    though this analogy is not apparent.

    - For example, the Ads team in the Google office building is a pod.
    - For example, the YouTube team in the Google office building is another
      pod.
    - For example, the Google Brain team in the Google office building is
      another pod.

4. **Containers**: These are the employees/workers within the offices (pods).
   Each worker (container) has a specific job to do. A pod (office) may have
   just one worker (container), or it may have many workers (containers) who
   work together to get the job done.

    - For example, two software engineers and one machine learning engineer in
      the Google Brain team in the Google office building.

5. **Deployment**: This is the **HR team's plan** for how to staff the offices.
   The plan might say that there should always be at least five employees
   working on Project A. If someone on Project A leaves the company, the HR team
   hires a new employee to fill that spot. Similarly, a Kubernetes Deployment
   checks that there are always the required number of pods running. If a pod
   crashes, the Deployment creates a new one to replace it.

6. **Services**: A service in Kubernetes is like a **department** in the
   business park. This department could be spread across multiple offices (pods)
   in different office buildings (nodes), but they all work towards a common
   goal. Similarly, a service in Kubernetes is a set of pods that work together,
   such as a backend for handling HTTP requests.

7. **Ingress**: Ingress can be seen as the **main gate** of the business park
   that controls and routes incoming traffic to the right office building (node)
   and office (pod).

8. **ConfigMaps and Secrets**: These are like **internal memos and confidential
   documents** respectively. ConfigMaps hold non-confidential data, like
   internal memos which are meant for everyone in the office to read and use. On
   the other hand, Secrets hold confidential data, similar to sensitive
   documents that should be only accessible to specific people.

9. **Volumes**: Volumes are like **filing cabinets** in an office. Containers,
   like employees, can store, modify, and retrieve data from them. However,
   unlike containers which can come and go, the filing cabinet (volume) remains,
   keeping the data intact.

## Kubernetes Cluster

### [Definition](https://kubernetes.io/docs/concepts/architecture/nodes/)

A Kubernetes cluster is made up of a set of nodes, which are the machines
(physical or virtual) where your workloads (containers) run.

So a Kubernetes cluster can have many nodes. A node is a worker machine and may
be either a virtual or a physical machine, depending on the cluster. Each node
contains the services necessary to run Pods and is managed by the master
components.

There are two types of nodes:

1. **Worker nodes**: These are the machines where the actual workloads
   (containers) run. They have Kubelet, which is an agent for managing the node
   and communicating with the Kubernetes master. They also have tools for
   handling container operations, like containerd or Docker.

2. **Master node**: This is the controlling machine that manages the worker
   nodes and the pods in the cluster. The master node communicates with the
   worker nodes via the Kubernetes API, which the master exposes. The master
   node schedules deployments and manages the desired state of the cluster.

A cluster with more than one node allows you to run your applications and
services in a high-availability setup, where one or more nodes can fail, and
your applications will still be available. It also enables you to scale out your
application or service by running more pods on more nodes.

### [What is a Kubelet?](https://kubernetes.io/docs/reference/command-line-tools-reference/kubelet/)

Kubelet is an essential component of Kubernetes that runs on each node (worker
and master) in the cluster. It's responsible for maintaining the desired state
for pods. Here's a bit more detail:

1. **Pod Management**: The Kubelet receives a PodSpec (a YAML or JSON object
   that describes a pod) from the API server and ensures that the containers
   described in the PodSpec are running and healthy. The PodSpecs are created as
   a result of the API server processing calls to the Kubernetes API.

2. **Node Registration**: When a Kubelet starts on a node, it registers the node
   with the API server providing information such as the node's IP address,
   hostname, and available resources (CPU, memory, etc).

3. **Health Checks**: Kubelet also performs health checks to ensure that the
   applications (containers) running in the pods are healthy. If a container
   becomes unresponsive, based on the policy set, Kubelet can restart the
   container to keep the application running.

4. **Resource Metrics**: Kubelet collects and reports resource metrics such as
   CPU, memory usage, etc., for the containers running on its node to the API
   server.

5. **Interaction with Container Runtime**: Kubelet interacts with the container
   runtime (Docker, containerd, CRI-O etc.) on the node to manage the lifecycle
   of the container like pulling the image, starting or stopping the container.

So, in essence, Kubelet acts as a bridge between the master and the nodes,
enabling the master to have control over various aspects of node and pod
management.

### What is the difference between a master node and a worker node?

- Master nodes, also called control plane nodes, are responsible for
    controlling and managing the overall operation of the Kubernetes cluster.
    They host components such as the API server, controller manager, scheduler,
    and etcd, which is the cluster database.
- Worker nodes, on the other hand, are where your applications (inside
    containers) actually run. They contain all the necessary services to manage
    the networking between the containers, communicate with the master node, and
    assign resources to the containers scheduled.

### How does the master node communicate with worker nodes?

The master node communicates with worker nodes through the Kubernetes API, which
is exposed by the API server on the master node. The Kubelet, running on each
worker node, communicates with the master node's API server. This way, the
master node can schedule and control the state of workloads on worker nodes.

### How does Kubernetes handle node failures to ensure high availability?

- Kubernetes uses various mechanisms to handle node failures. These include
    replication of application instances across different nodes, regular health
    checks, and automated pod rescheduling.
- For example, if a worker node fails, the Replication Controller notices the
    drop in the number of replicas, because the pods running on the failed node
    are no longer reachable. The Replication Controller then creates new pods on
    other available nodes to maintain the desired number of replicas.
- Similarly, the master nodes in a Kubernetes cluster can also be set up in a
    highly available configuration to ensure that the failure of a single master
    node doesn't bring down the entire cluster.

## [Kubernetes Node](https://kubernetes.io/docs/concepts/architecture/nodes/)

### Definition

A node is a worker machine in Kubernetes, which may be either a virtual or a
physical machine, depending on the cluster. Each node contains the services
necessary to run Pods and is managed by the master components. There are two
types of nodes:

1. **Worker nodes**: Machines where actual workloads (containers) run. They have
   Kubelet, an agent for managing the node and communicating with the Kubernetes
   master. They also have tools for handling container operations, like
   containerd or Docker.

2. **Master nodes**: The controlling machines that manage the worker nodes and
   the pods in the cluster. Master nodes communicate with worker nodes via the
   Kubernetes API, which the master node exposes. Master nodes schedule
   deployments and manage the desired state of the cluster.

### Node Status

Each node has a status that contains information about its capacity (CPU,
memory, disk space), its conditions (Disk Pressure, Memory Pressure, PID
Pressure, Ready), and its addresses (Host name, External IP, Internal IP).

You can get the status of a node by running the following command:

```bash
kubectl get nodes
```

### Node Lifecycle

Kubernetes maintains the lifecycle of a node, from when it is first registered
with the cluster, through its ongoing maintenance, to its eventual retirement or
failure. Node Controller is the component responsible for noticing and
responding when nodes go down.

### [Pods](https://kubernetes.io/docs/concepts/workloads/pods/)

### Definition

A Pod (as in a pod of whales or pea pod) is a group of one or more
[containers](https://kubernetes.io/docs/concepts/containers/), with shared
storage and network resources, and a specification for how to run the
containers. A Pod's contents are always co-located and co-scheduled, and run in
a shared context. A Pod models an application-specific "logical host": it
contains one or more application containers which are relatively tightly
coupled. In non-cloud contexts, applications executed on the same physical or
virtual machine are analogous to cloud applications executed on the same logical
host.

### What is a Pod?

> The shared context of a Pod is a set of Linux namespaces, cgroups, and
> potentially other facets of isolation - the same things that isolate a
> container. Within a Pod's context, the individual applications may have
> further sub-isolations applied.

> A Pod is similar to a set of containers with shared namespaces and shared
> filesystem volumes.

Pods are the smallest deployable units in Kubernetes that can be created and
managed. Each Pod represents a single instance of a running process in the
cluster and can contain one or more containers. Containers within a Pod share an
IP address and port space, and can communicate with one another using
`localhost`. They can also share storage volumes.

---

**Containers**:
[Containers](https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/#why-containers)
are a technology for packaging the (compiled) code for an application along with
the dependencies it needs at run time. Each container that you run is
repeatable; the standardization from having dependencies included means you get
the same behavior wherever you run it.

### Common Interview Questions on Nodes, Pods, and Containers in Kubernetes

1. **What is the relationship between a Pod, a Node, and a Container in
   Kubernetes?**

    In Kubernetes, a **Node** is a worker machine and it may be either a virtual
    or a physical machine. Each Node is managed by the Master and is configured
    with multiple components to manage the workloads.

    A **Pod** is the smallest and simplest deployable unit of computing that you
    can create and manage in Kubernetes. A Pod encapsulates an application's
    container (or, in some cases, multiple containers), storage resources, a
    unique network IP, and options that govern how the container(s) should run.
    A Pod is hosted inside a Node and multiple Pods can run on the same Node.

    **Containers** are a good way to bundle and run your applications. A
    container runs natively on Linux and shares the kernel of the host machine
    with other containers. It runs a discrete process, taking no more memory
    than any other executable, making it lightweight. The nature of a container
    allows it to be easily packaged and shipped with its dependencies, which is
    encapsulated into a Kubernetes Pod.

2. **What happens when a node in a Kubernetes cluster fails?**

    When a node in a Kubernetes cluster fails, the
    [Kubernetes master](https://kubernetes.io/docs/concepts/architecture/control-plane-node-communication/#control-plane-to-node)
    will automatically detect this event. Once the Node Controller notices that
    a node has not reported back for a specified amount of time (the default is
    5 minutes), it will declare the node as unreachable. The workloads (Pods)
    running on that node will be scheduled for deletion. The Replication
    Controllers in the Kubernetes master will then take care of creating new
    pods and scheduling them onto the healthy nodes in the cluster, ensuring
    that the desired state of the application is maintained and providing high
    availability.

3. **What is the role of Kubelet in a Kubernetes cluster?**

    [Kubelet](https://kubernetes.io/docs/concepts/overview/components/#kubelet)
    is a key component of a Kubernetes Node. It's responsible for ensuring that
    the containers in a Pod are running and in a healthy state. It maintains the
    desired state for pods, registers the node with the API server, performs
    health checks, collects resource metrics, and interacts with the container
    runtime on the node.

These are just some of the questions that can come up during an interview about
Kubernetes. Remember, it's important to understand the principles behind the
technology in order to answer questions effectively.

### What's a Pod?

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
