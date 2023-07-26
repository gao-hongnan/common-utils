# Kubernetes

- [Kubernetes](#kubernetes)
  - [The Analogies](#the-analogies)
  - [Kubernetes Cluster](#kubernetes-cluster)
    - [Definition](#definition)
    - [What is the difference between a master node and a worker node?](#what-is-the-difference-between-a-master-node-and-a-worker-node)
    - [How does the master node communicate with worker nodes?](#how-does-the-master-node-communicate-with-worker-nodes)
    - [How does Kubernetes handle node failures to ensure high availability?](#how-does-kubernetes-handle-node-failures-to-ensure-high-availability)
  - [Kubernetes Node](#kubernetes-node)
    - [Definition](#definition-1)
    - [Node Status](#node-status)
    - [Node Lifecycle](#node-lifecycle)
    - [What is a Kubelet?](#what-is-a-kubelet)
    - [What happens if a node in Kubernetes fails?](#what-happens-if-a-node-in-kubernetes-fails)
  - [Pods](#pods)
    - [Definition](#definition-2)
    - [What is a Pod?](#what-is-a-pod)
    - [How does a Pod differ from a Docker container?](#how-does-a-pod-differ-from-a-docker-container)
    - [What is the lifecycle of a Pod in Kubernetes?](#what-is-the-lifecycle-of-a-pod-in-kubernetes)
    - [What are Init Containers and how are they different from regular containers in a Pod?](#what-are-init-containers-and-how-are-they-different-from-regular-containers-in-a-pod)
    - [Explain the role of a sidecar container in a Pod](#explain-the-role-of-a-sidecar-container-in-a-pod)
    - [What are multi-container pods and why might you use them?](#what-are-multi-container-pods-and-why-might-you-use-them)
    - [How do you share data between containers in a Pod?](#how-do-you-share-data-between-containers-in-a-pod)
    - [How do you control what nodes a Pod can be scheduled on?](#how-do-you-control-what-nodes-a-pod-can-be-scheduled-on)
    - [How would you troubleshoot a crashing Pod?](#how-would-you-troubleshoot-a-crashing-pod)
    - [Explain the difference between a liveness probe and a readiness probe](#explain-the-difference-between-a-liveness-probe-and-a-readiness-probe)
    - [How can you update the containers in a running Pod?](#how-can-you-update-the-containers-in-a-running-pod)
    - [What is the difference between an emptyDir volume and a hostPath volume?](#what-is-the-difference-between-an-emptydir-volume-and-a-hostpath-volume)
  - [Containers](#containers)
  - [Deployments](#deployments)
    - [Definition](#definition-3)
    - [How do Deployments work?](#how-do-deployments-work)
    - [Rolling Updates and Rollbacks](#rolling-updates-and-rollbacks)
    - [Scaling Deployments](#scaling-deployments)
    - [What is the purpose of a Deployment in Kubernetes?](#what-is-the-purpose-of-a-deployment-in-kubernetes)
    - [How do Deployments handle updates?](#how-do-deployments-handle-updates)
    - [How can you rollback a Deployment?](#how-can-you-rollback-a-deployment)
    - [What are the strategies for Deployment updates in Kubernetes?](#what-are-the-strategies-for-deployment-updates-in-kubernetes)
  - [ReplicaSets](#replicasets)
    - [Definition](#definition-4)
    - [How do ReplicaSets work?](#how-do-replicasets-work)
    - [When to use a ReplicaSet?](#when-to-use-a-replicaset)
    - [What is the difference between a ReplicaSet and a Deployment?](#what-is-the-difference-between-a-replicaset-and-a-deployment)
    - [How does a ReplicaSet know what Pods to manage?](#how-does-a-replicaset-know-what-pods-to-manage)
    - [What does it mean to have N Replicas?](#what-does-it-mean-to-have-n-replicas)
    - [What happens when a Pod in a ReplicaSet fails?](#what-happens-when-a-pod-in-a-replicaset-fails)
    - [How do you scale a ReplicaSet?](#how-do-you-scale-a-replicaset)
    - [Job](#job)
    - [Definition](#definition-5)
    - [How Jobs Work](#how-jobs-work)
    - [Job Lifecycle](#job-lifecycle)
  - [CronJob](#cronjob)
    - [Definition](#definition-6)
    - [How CronJobs Work](#how-cronjobs-work)
    - [CronJob Lifecycle](#cronjob-lifecycle)
  - [Services, Load Balancing, and Networking](#services-load-balancing-and-networking)
    - [Service](#service)
      - [Definition](#definition-7)
      - [Types of Services](#types-of-services)
        - [ClusterIP](#clusterip)
        - [NodePort](#nodeport)
        - [LoadBalancer](#loadbalancer)
        - [ExternalName](#externalname)
      - [Why is Service Important?](#why-is-service-important)
      - [Can a Deployment exist without a Service?](#can-a-deployment-exist-without-a-service)
  - [Storage](#storage)
    - [Volumes](#volumes)
      - [Definition](#definition-8)
      - [What is a Volume?](#what-is-a-volume)
    - [Persistent Volumes (PVs)](#persistent-volumes-pvs)
      - [Definition](#definition-9)
      - [Interview Questions on Persistent Volumes](#interview-questions-on-persistent-volumes)
    - [Persistent Volume Claims (PVCs)](#persistent-volume-claims-pvcs)
      - [Definition](#definition-10)
      - [Interview Questions on Persistent Volume Claims](#interview-questions-on-persistent-volume-claims)
  - [Debugging](#debugging)
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

## [Kubernetes Cluster](https://kubernetes.io/docs/concepts/architecture/nodes/)

### Definition

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

You can retrieve the status of a node by executing the following command:

```bash
kubectl get nodes
```

### Node Lifecycle

Kubernetes maintains the lifecycle of a node, from when it is first registered
with the cluster, through its ongoing maintenance, to its eventual retirement or
failure. Node Controller is the component responsible for noticing and
responding when nodes go down.

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

### What happens if a node in Kubernetes fails?

If a pod fails, the `ReplicaSet` ensures that it's restarted to maintain the
desired number of pods. However, when a node fails, the process is different.

If a Kubernetes node fails, the Node Controller in the Kubernetes control plane
notices the node's absence. All pods running on the failed node are marked as
'Lost' and scheduled for deletion. The workload of these pods, if governed by
controllers like ReplicaSets, Deployments, or StatefulSets, will be shifted and
recreated on other available nodes.

Kubernetes itself doesn't handle the restart of nodes as that responsibility
lies more with the underlying infrastructure. The process to handle a node
restart or replacement is dependent on the setup of the cluster. For example, if
your Kubernetes cluster is running in a cloud provider environment, the cloud
provider usually handles the replacement of the failed node, depending on your
configured settings. In an on-premises environment, you may need to manually
replace or fix the failed node.

In essence, Kubernetes handles the scheduling and lifecycle of pods. The
lifecycle of nodes, on the other hand, is usually handled by the infrastructure
that Kubernetes is deployed on.

## [Pods](https://kubernetes.io/docs/concepts/workloads/pods/)

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

### How does a Pod differ from a Docker container?

While a Docker container encapsulates an application and its dependencies, a
Kubernetes Pod wraps one or more such containers (usually, containerized
applications), along with shared storage/network, and a specification for how to
run the containers.

### What is the lifecycle of a Pod in Kubernetes?

A Pod in Kubernetes goes through several phases during its lifecycle, including
Pending, Running, Succeeded, Failed, and Unknown. Understanding these states and
the transitions between them can help you manage and troubleshoot Pods more
effectively.

### What are Init Containers and how are they different from regular containers in a Pod?

Init Containers are a special type of container that run before the
application's main container. They are used to set up the right environment for
the main container to run effectively. Once the task of the Init Container is
complete, it exits, and the main container starts.

### Explain the role of a sidecar container in a Pod

Sidecar containers in Kubernetes are utility containers that support the main
container in a Pod. Examples of sidecar pattern use cases include log or data
change watchers, monitoring adapters, and proxies or bridges to different
network topologies.

### What are multi-container pods and why might you use them?

A multi-container Pod in Kubernetes is a Pod that hosts multiple containers
which work together to provide a service. The containers in such a Pod are
guaranteed to be co-located on the same machine and can share resources, making
them ideal for cooperative tasks.

### How do you share data between containers in a Pod?

Data sharing between containers within a Pod in Kubernetes is achieved using
shared volumes. A volume represents a storage area that is accessible to all
containers running in a Pod, and exists for as long as the Pod exists.

### How do you control what nodes a Pod can be scheduled on?

You can control Pod scheduling using several mechanisms in Kubernetes, including
node selectors, node affinity and anti-affinity rules, and taints and
tolerations. These mechanisms allow you to influence where your Pods are
scheduled based on factors like node labels, resource availability, or other
custom rules.

### How would you troubleshoot a crashing Pod?

Troubleshooting a crashing Pod in Kubernetes involves a set of methodologies and
commands, like `kubectl describe pod` to get detailed information about the Pod,
`kubectl logs` to fetch the logs of the Pod, and checking the Events associated
with the Pod.

### Explain the difference between a liveness probe and a readiness probe

In Kubernetes, liveness and readiness probes are used to manage the health of
containers. A liveness probe is used to know when to restart a container, while
a readiness probe is used to know when a container is ready to start accepting
traffic.

### How can you update the containers in a running Pod?

Containers in a running Pod cannot be directly updated due to the immutable
nature of Pods. Instead, you update the Deployment configuration that manages
the Pod. The Deployment creates a new ReplicaSet based on the updated
configuration, and then gradually replaces the existing Pods with new ones.

### What is the difference between an emptyDir volume and a hostPath volume?

Both emptyDir and hostPath are types of volumes in Kubernetes, used for
different purposes. An emptyDir volume is first created when a Pod is assigned
to a Node, and exists as long as that Pod is running on that node, while a
hostPath volume mounts a file or directory from the host node's filesystem into
your Pod. Understanding the differences and use cases for these volume types is
crucial for managing storage needs in Pods.

## [Containers](https://kubernetes.io/docs/concepts/containers/)

Containers are a technology for packaging the (compiled) code for an application
along with the dependencies it needs at run time. Each container that you run is
repeatable; the standardization from having dependencies included means you get
the same behavior wherever you run it.

One should be clear of what container is if you are familiar with
containerization technology such as Docker.

> Each node in a Kubernetes cluster runs the containers that form the Pods
> assigned to that node. Containers in a Pod are co-located and co-scheduled to
> run on the same node.

Absolutely, here are the sections for Deployments and ReplicaSets:

## [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

### Definition

A Deployment in Kubernetes is a higher-level concept that manages ReplicaSets
and provides declarative updates to Pods along with a lot of other features.
This means that a Deployment will ensure that a certain number of identical Pods
are running and available at all times.

### How do Deployments work?

Deployments work by creating a ReplicaSet for each new Deployment revision. The
Deployment will then gradually replace Pods from the old ReplicaSet with Pods
from the new ReplicaSet to achieve the desired state.

### Rolling Updates and Rollbacks

Deployments support updating images (or any other field in the Pod template)
through rolling updates. Rolling updates gradually replace the old Pods by the
new ones. If a Deployment is not going as planned, you can pause it and then
rollback to a previous revision.

### Scaling Deployments

Deployments can be scaled manually by modifying the number of replicas, or
automatically based on CPU usage through Horizontal Pod Autoscalers.

### What is the purpose of a Deployment in Kubernetes?

Deployments in Kubernetes are used to manage stateless applications, providing
declarative updates, scalability, and rollback functionalities.

### How do Deployments handle updates?

Deployments manage updates by creating a new ReplicaSet and increasing the
number of replicas while the old ones decrease. This achieves a rolling update,
ensuring zero downtime deployments.

### How can you rollback a Deployment?

A Deployment can be rolled back to a previous revision using the
`kubectl rollout undo` command followed by the deployment name and the revision
number.

### What are the strategies for Deployment updates in Kubernetes?

Kubernetes Deployments support two strategies for updates - Recreate and Rolling
Update. In the Recreate strategy, all old Pods are killed before new ones are
created. In the Rolling Update strategy, the Deployment updates Pods in a
rolling fashion, ensuring that service is not disrupted during the update.

## [ReplicaSets](https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/)

### Definition

A ReplicaSet in Kubernetes is a lower-level form of Deployment that ensures a
specified number of pod replicas are running at any given time. Unlike
Deployments, ReplicaSets do not support rolling updates, and are not recommended
to be used directly.

### How do ReplicaSets work?

A ReplicaSet is linked to its Pods via the Pods' metadata.ownerReferences field,
which specifies what resource the current object is owned by. The ReplicaSet
uses label selectors to determine what Pods fall under its responsibility.

### When to use a ReplicaSet?

ReplicaSets are typically used when you always want a static number of Pods
running for a particular application. However, they are often used indirectly
through Deployments, which are a higher-level concept that manage ReplicaSets
and provide declarative updates to Pods along with many other useful features.

### What is the difference between a ReplicaSet and a Deployment?

The primary difference between a ReplicaSet and a Deployment is that Deployments
provide a declarative way to update the Pods using rolling updates, while
ReplicaSets do not.

The `replicas` field in the `Deployment` YAML directly corresponds to the
desired number of replicas for the `ReplicaSet` that the `Deployment` manages.

Here's a bit more context:

- A `Deployment` in Kubernetes is a higher-level concept that manages
    `ReplicaSets`, and describes a desired state for your application. It
    handles updates to your application in a controlled way.
- The `ReplicaSet` is a lower-level concept, and its primary purpose is to
    maintain a stable set of replica pods running at any given time. This is
    where the `replicas` field in the `Deployment` specification comes into
    play.
- When you specify a `Deployment`, you specify the number of replicas of a pod
    that you want running. This `replicas` field in the `Deployment`
    configuration is translated into the desired number of replicas in the
    `ReplicaSet` that the `Deployment` creates and manages.
- Kubernetes will then try to match the actual state (the number of running
    pods) to the desired state (the `replicas` number specified in the
    `Deployment` and thus the `ReplicaSet`).

In summary, the `replicas` field you set in a `Deployment` will be the number of
replicas set for the `ReplicaSet` that the `Deployment` controls.

### How does a ReplicaSet know what Pods to manage?

A ReplicaSet knows what Pods to manage using label selectors. The labels of the
Pods and the selectors of the ReplicaSet must match for the Pods to fall under
the control of the ReplicaSet.

### What does it mean to have N Replicas?

When you create a Deployment or a ReplicaSet with a `replicas` count of 3,
Kubernetes will launch three identical pods. Each of these pods will be running
the same container(s) based on the same Docker image(s), and they'll be
configured the same way. This is often used to provide load balancing and fault
tolerance for your application.

In the case of a model deployment, each pod would be running a copy of the same
model. The pods are independent of each other and don't share state, so they can
all handle requests simultaneously.

When you create a Service to expose your Deployment or ReplicaSet, the Service
will automatically load balance traffic between the available pods. This means
that incoming requests will be spread out across all three pods. If one pod
becomes unavailable (due to a crash, for example), the Service will
automatically route traffic to the remaining pods until the failed pod is
replaced.

Remember that while the pods are identical, the data they process could be
different, depending on the incoming requests they receive. Each pod can handle
different requests independently of the others.

### What happens when a Pod in a ReplicaSet fails?

If a Pod in a ReplicaSet fails, the ReplicaSet will automatically create a new
Pod to replace it and maintain the desired count.

So if your ReplicaSet has a `replicas` count of 3, and one of the Pods fails,
the ReplicaSet will create a new Pod to replace it, and you'll still have 3 Pods
running.

### How do you scale a ReplicaSet?

A ReplicaSet can be scaled manually by modifying the number of replicas in the
ReplicaSet configuration. The ReplicaSet will then create or delete Pods as
needed to reach the desired number. However, for automatic scaling based on CPU
usage or other metrics, you'd typically use a Horizontal Pod Autoscaler, which
operates on a Deployment.

### [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/)

### Definition

A **job** in Kubernetes is a controller object that represents a finite task or
a compute task that needs to run to completion. Jobs differ from other
controller objects in that Jobs manage a task as it runs over time to
completion, rather than managing a desired state such as the total number of
running pods.

Jobs are most commonly used in applications that need to run batch processes or
perform a computation and then terminate.

### How Jobs Work

When a Job is created, it creates one or more Pods and ensures that a specified
number of them successfully terminate. As pods successfully complete, the Job
tracks the successful completions. When a specified number of successful
completions is reached, the Job itself is complete.

Jobs can also be set to run multiple pods in parallel, run multiple pods
sequentially, or even run pods with a specified number of failures.

### Job Lifecycle

Typically, a Job completes when the specified number of Pods have successfully
completed. At this point, no new Pods are created by the Job, but existing Pods
are not deleted either. Keeping them around allows you to still view the logs of
completed pods to check for errors, warnings, or other diagnostic output. The
Job object also remains after it is completed so that you can view its status.
It is up to the user to delete old jobs after noting their status.

## [CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)

### Definition

A `CronJob` is a controller object in Kubernetes that runs Jobs on a time-based
schedule, much like the `cron` utility in Unix-based systems. It is commonly
used for running periodic tasks, such as backups, report generation, or sending
emails.

A `CronJob` is a `Job`.

### How CronJobs Work

CronJobs manage time-based Jobs, that is, they can be scheduled to run at
specific points in time or at regular intervals. A CronJob object is like a line
in a crontab (cron table) file. It runs a Job periodically on a given schedule,
written in Cron format.

CronJobs are useful for creating periodic and recurring tasks, like running
backups or sending emails. They can also schedule individual tasks for a
specific time, such as scheduling a Job for when your cluster is likely to be
idle.

### CronJob Lifecycle

A CronJob creates Jobs on a repeating schedule according to the `.spec.schedule`
field of the CronJob object, as well as the `.spec.jobTemplate` of the Job that
it should run.

Failed Jobs created by a CronJob are not retried by the CronJob. If the
`.spec.jobTemplate.spec.template.spec.restartPolicy` field is not set to
`OnFailure` or `Always`, it defaults to `Never`. You can check the status of the
CronJob using the `kubectl describe cronjob <cronjob-name>` command.

It's worth noting that if the CronJob controller is down when a Job should have
been started, that Job is not run. Also, the CronJob controller only checks for
missed schedules within a look-back window, which by default is only a minute
long. This means that if the controller is down for longer than this look-back
window, no checks for missed Jobs will occur.

## Services, Load Balancing, and Networking

### [Service](https://kubernetes.io/docs/concepts/services-networking/service/)

#### Definition

> In Kubernetes, a Service is a method for exposing a network application that
> is running as one or more Pods in your cluster.

A key aim of Services in Kubernetes is that you don't need to modify your
existing application to use an unfamiliar service discovery mechanism. You can
run code in Pods, whether this is a code designed for a cloud-native world, or
an older app you've containerized. You use a Service to make that set of Pods
available on the network so that clients can interact with it.

If you use a Deployment to run your app, that Deployment can create and destroy
Pods dynamically. From one moment to the next, you don't know how many of those
Pods are working and healthy; you might not even know what those healthy Pods
are named. Kubernetes Pods are created and destroyed to match the desired state
of your cluster. Pods are ephemeral resources (you should not expect that an
individual Pod is reliable and durable).

Each Pod gets its own IP address (Kubernetes expects network plugins to ensure
this). For a given Deployment in your cluster, the set of Pods running in one
moment in time could be different from the set of Pods running that application
a moment later.

This leads to a problem: if some set of Pods (call them "backends") provides
functionality to other Pods (call them "frontends") inside your cluster, how do
the frontends find out and keep track of which IP address to connect to, so that
the frontend can use the backend part of the workload?

Enter Services.

#### Types of Services

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

Each Service has a `type` field which defines the type of Service it is. The
type dictates how the Service is exposed to network traffic.

There are different types of Services in Kubernetes.

##### ClusterIP

Exposes the Service on a cluster-internal IP. This makes the Service only
reachable from within the cluster. This is the default ServiceType.

##### NodePort

Exposes the Service on each Node’s IP at a static port (the NodePort). A
ClusterIP Service, to which the NodePort Service routes, is automatically
created.

##### LoadBalancer

Exposes the Service externally using a cloud provider’s load balancer. NodePort
and ClusterIP Services, to which the external load balancer routes, are
automatically created.

##### ExternalName

Maps the Service to the contents of the `externalName` field (e.g.
`foo.bar.example.com`), by returning a CNAME record.

#### Why is Service Important?

A Service in Kubernetes is crucial when you're dealing with dynamic Pod creation
and destruction. When a Pod dies and gets replaced, it will have a different IP,
but the Service ensures that the traffic will always get to the intended
destination.

#### Can a Deployment exist without a Service?

Yes. A Deployment can exist without a Service in Kubernetes. However, without a
Service, a Deployment won't be accessible from the network, whether it be from
within the cluster or from outside.

The Pods deployed would only be reachable by their IP addresses within the
cluster, and those IP addresses are typically ephemeral - they change whenever
the Pod is recreated, which can happen for a number of reasons including scaling
operations, updates, and node failures.

Services provide a stable endpoint by which other entities, including other
Pods, can communicate with the Pods that make up the Deployment. This is
especially important if you're exposing an application or API to the outside
world, but it's also useful for internal communications within the cluster.

## Storage

### [Volumes](https://kubernetes.io/docs/concepts/storage/volumes/)

#### Definition

On-disk files in a container are ephemeral, which presents some problems for non-trivial applications when running in containers. One problem occurs when a container crashes or is stopped. Container state is not saved so all of the files that were created or modified during the lifetime of the container are lost. During a crash, kubelet restarts the container with a clean state. Another problem occurs when multiple containers are running in a Pod and need to share files. It can be challenging to setup and access a shared filesystem across all of the containers. The Kubernetes volume abstraction solves both of these problems. Familiarity with Pods is suggested.



In Kubernetes, a Volume is a directory, possibly with some data in it, which is accessible to a Container within a Pod. It's a mechanism for persisting data generated by and used by Docker containers that are part of a Pod. Unlike a Docker volume, a Kubernetes Volume has an explicit lifetime—the same as the Pod that encloses it.

When a Pod ceases to exist, the Kubernetes volume also ceases to exist. It’s designed this way to support the temporary storage use case. This approach handles the issue of data persistence across container restarts.

Kubernetes supports many types of Volumes: `emptyDir`, `hostPath`, `awsElasticBlockStore`, `gcePersistentDisk`, `nfs`, `iscsi`, etc.



#### What is a Volume?

Kubernetes supports many types of volumes. A Pod can use any number of volume types simultaneously. Ephemeral volume types have a lifetime of a pod, but persistent volumes exist beyond the lifetime of a pod. When a pod ceases to exist, Kubernetes destroys ephemeral volumes; however, Kubernetes does not destroy persistent volumes. For any kind of volume in a given pod, data is preserved across container restarts.

At its core, a volume is a directory, possibly with some data in it, which is accessible to the containers in a pod. How that directory comes to be, the medium that backs it, and the contents of it are determined by the particular volume type used.

To use a volume, specify the volumes to provide for the Pod in .spec.volumes and declare where to mount those volumes into containers in .spec.containers[*].volumeMounts. A process in a container sees a filesystem view composed from the initial contents of the container image, plus volumes (if defined) mounted inside the container. The process sees a root filesystem that initially matches the contents of the container image. Any writes to within that filesystem hierarchy, if allowed, affect what that process views when it performs a subsequent filesystem access. Volumes mount at the specified paths within the image. For each container defined within a Pod, you must independently specify where to mount each volume that the container uses.

Volumes cannot mount within other volumes (but see Using subPath for a related mechanism). Also, a volume cannot contain a hard link to anything in a different volume.



### [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) (PVs)

#### Definition

A Persistent Volume (PV) in Kubernetes is a piece of storage in the cluster that has been provisioned by an administrator or dynamically provisioned using Storage Classes. It is a resource in the cluster just like a node is a cluster resource. PVs are volume plugins like Volumes but have a lifecycle independent of any individual Pod that uses the PV. This API object captures the details of the implementation of the storage, be that NFS, iSCSI, or a cloud-provider-specific storage system.

#### Interview Questions on Persistent Volumes

**Q1:** What is a Persistent Volume (PV) in Kubernetes?

**Q2:** How is a PV different from a regular Kubernetes Volume?

**Q3:** What are the two types of PVs?

---

### Persistent Volume Claims (PVCs)

#### Definition

A Persistent Volume Claim (PVC) is a request for storage by a user. It is similar to a Pod. Pods consume node resources and PVCs consume PV resources. Pods can request specific levels of resources (CPU and Memory). Claims can request specific size and access modes (e.g., they can be mounted ReadWriteOnce, ReadOnlyMany, or ReadWriteMany).

Once a user has a claim and that claim is bound to a volume, the bound PV belongs to the user for as long as she needs it. Users schedule Pods and access their claimed PVs by including a PersistentVolumeClaim in their Pod’s Volumes block.

#### Interview Questions on Persistent Volume Claims

**Q1:** What is a Persistent Volume Claim (PVC)?

**Q2:** How does a PVC interact with a PV?

**Q3:** How do Pods use PVCs to access storage resources?

**Q4:** What are some access modes for PVCs?

**Q5:** How do PVCs and PVs clean up when they are no longer needed?

**Q6:** How does storage resizing work in Kubernetes?


## Debugging

`kubectl` offers several commands to help you understand the state of your
cluster and debug any issues. Here's a brief overview of some of these commands:

1. `kubectl logs`: This command is used to print the logs from a container in a
   pod. If the pod has multiple containers, you need to specify the container to
   get its logs. This command is useful when you want to understand what's
   happening within your application at runtime or debug any runtime issues. For
   instance, any exceptions or errors that your application throws will be
   present in these logs.

2. `kubectl describe`: This command gives detailed information about a resource
   (like a pod, deployment, service, etc.). It includes things like the current
   state of the resource, recent events, and metadata. It can help you
   understand the current state of your resource and track any recent changes.

3. `kubectl get`: This command lists one or more resources. This can give you a
   quick overview of the resources currently in your cluster.

4. `kubectl exec`: This command is used to run commands in a container. This can
   be very useful for debugging, as it allows you to inspect the container's
   filesystem, check the running processes, etc.

5. `kubectl diff`: This command helps to find differences between the current
   live state and the configuration specified in files or other resources.

The best command to use depends on what you're trying to achieve or the issue
you're trying to debug. You might use `kubectl logs` if you're debugging an
application issue, `kubectl describe` if you're trying to understand why a pod
isn't starting, or `kubectl exec` if you're trying to explore the state of a
running container.

Here's a structured and logical sequence of steps you can follow to debug a
failing Kubernetes application:

1. **Identify the problem:** First, you should understand the symptoms of the
   issue. Is your application not accessible? Is it not responding as expected?
   Is there an outage?

2. **Isolate the component:** Once you have identified the problem, determine
   the component that is causing the issue. This could be the service, the
   deployment, the pod, or even the node where the pod is running.

3. **Check the Kubernetes Objects status:**

    - **Pods**: Use the command `kubectl get pods`. If any pod is not in the
      "Running" state, it might be the source of the issue.

    - **Nodes**: Check the status of the nodes with `kubectl get nodes`. If a
      node is in the "NotReady" state, the pods running on it might be affected.

    - **Services**: Check the services with `kubectl get services`. If your
      application is not accessible, the issue might be with the service.

4. **Check the Events:** Use `kubectl get events` to check for any abnormal
   events in the namespace.

5. **Inspect the Pod:** Use `kubectl describe pod <pod-name>` for a failing pod.
   This command provides more detailed information about the pod and any events
   or errors associated with it.

6. **Inspect the Pod logs:** Use `kubectl logs <pod-name>` to check the logs of
   the application running in the pod. This can give you more detailed error
   messages about the application-level issues.

7. **Verify Resource Availability:** Ensure that your cluster has sufficient
   resources (CPU, memory, storage). If you're running out of any of these, it
   could cause pods to crash or be evicted.

8. **Check Configuration Files:** If all the above steps do not point to a clear
   issue, check your Kubernetes configuration files (YAML manifests) for any
   misconfigurations.

9. **Inspect Application Code:** If all else fails, the problem may lie with the
   application code itself. Debug the application outside of Kubernetes, if
   possible, to see if the issue persists.

10. **Use Debugging Tools:** Use Kubernetes debugging and validation tools to
    get more insights if the problem remains unresolved.
    1. `kubeval` etc.

Remember, the sequence might vary slightly depending on the nature of the
problem. The general idea is to start from the high-level Kubernetes objects and
go deeper until the root cause is found.

```bash
kubectl exec -it fastapi-server-deployment-5bf55f9f76-262nr -- /bin/bash
```

## Summary

In Kubernetes, a **Node** is a worker machine and it may be either a virtual or
a physical machine. Each Node is managed by the Master and is configured with
multiple components to manage the workloads.

A **Pod** is the smallest and simplest deployable unit of computing that you can
create and manage in Kubernetes. A Pod encapsulates an application's container
(or, in some cases, multiple containers), storage resources, a unique network
IP, and options that govern how the container(s) should run. A Pod is hosted
inside a Node and multiple Pods can run on the same Node.

**Containers** are a good way to bundle and run your applications. A container
runs natively on Linux and shares the kernel of the host machine with other
containers. It runs a discrete process, taking no more memory than any other
executable, making it lightweight. The nature of a container allows it to be
easily packaged and shipped with its dependencies, which is encapsulated into a
Kubernetes Pod.

MLFLOPW LOGS
