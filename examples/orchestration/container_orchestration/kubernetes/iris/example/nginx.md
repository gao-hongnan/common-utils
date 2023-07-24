# NGINX

## Deployment

This deployment create one pod which has only one container `nginx` (you can create more than one under spec/containers).
The `containerPort` is 80 means the container will listen on port 80 - runs a process is alive and will be
can serve requests that comes to this port.

then you can label this pod to be `app: nginx` under template/metadata.

> so far created 1 pod with 1 container and is labelled key app value nginx.
> For eg, mlflow dashboard is a webserver.


```bash
kubectl apply -f deployment.yaml
```

You see 3 replicas.

kubectl port-forward deployment/my-nginx-deployment 8080:80

```
  selector:
    matchLabels:
      app: nginx
```

this chunk defines which pods go under this deployment?

This deployment object cannot serve anything but for eg, you can dockerize the training
script and run as a deployment object. but if you want to serve the predictions then you need service.

## Service

If you've created a `Service` of type `NodePort`, Kubernetes will allocate a port from a range (default is 30000-32767) on each Node for this Service, and any connection to this port on any Node is forwarded to the service.

After applying the `Service` configuration with `kubectl apply -f <filename>`, you can find out what port was allocated by describing the service:

```bash
kubectl describe service my-nginx-service
```

Look for the `NodePort:` line under `Ports:`. The number after `NodePort:` is the port you can use on your local machine to access the service.

If you're using a local Kubernetes solution like Minikube or Docker Desktop, you can access the service via `localhost:<NodePort>`. If you're using a cloud-based Kubernetes service, you'll need to use the public IP address of one of your nodes along with the NodePort to access the service.

Also, note that depending on your firewall and security settings, you may need to open this NodePort in your firewall to allow traffic to the service.

## Deployment vs Service

Services and Deployments in Kubernetes serve different but complementary roles. They're related in the sense that a Service is often used to provide network access to one or more Pods managed by a Deployment. Here's a brief overview of each:

- **Deployment**: A Deployment in Kubernetes is used to manage a set of identical Pods (the smallest and simplest unit in the Kubernetes object model that you create or deploy). A Deployment ensures that a specified number of Pods are running at all times. If a Pod goes down, the Deployment will start a new one. This is particularly useful for running stateless applications that can be scaled up and down easily.

- **Service**: While Deployments manage Pods and ensure their availability, they do not provide a stable network interface to these Pods as the Pod IPs are not guaranteed to stay the same over time. This is where Services come in. A Service in Kubernetes is an abstraction which defines a logical set of Pods and a policy by which to access them (sometimes this pattern is called a micro-service). Services enable the discovery of Pods and offer a stable IP address and DNS name by which Pods can be accessed.

So, while you technically can have Deployments without Services, any real-world application would require some way for users or other applications to interact with those Pods, and that's why you need Services.

To give you a real-world analogy: think of a Deployment like a corporation's workforce. The corporation can hire or fire employees (like a Deployment can create or destroy Pods), but it doesn't provide a way for the public to interact with them. Now, consider a Service as the corporation's customer service hotline. It provides a stable interface (the phone number) for the public to reach the corporation's services. This hotline can be routed to any available employee (like a Service can route traffic to any available Pod). Even if an employee leaves (or a Pod dies), the hotline still works and calls will be handled by other employees (other Pods).

## More on it

Let's break down each of the Kubernetes objects.

The **Deployment** you've defined is instructing Kubernetes to ensure that there are always three replicas (instances) of a Pod running the nginx image on your cluster. Here's a more detailed breakdown:

- `apiVersion: apps/v1`: This is the API version you're using to create the Deployment.
- `kind: Deployment`: This tells Kubernetes you're creating a Deployment.
- `metadata`: This section includes data about your Deployment, like its name.
- `spec`: This is where you define the behavior of the Deployment.
    - `replicas: 3`: This tells Kubernetes to keep three replicas of the Pod running at all times.
    - `selector`: This is how the Deployment identifies which Pods to manage. In this case, it's looking for Pods with the label `app: nginx`.
    - `template`: This defines the Pods that will be created. The Pods will have a label `app: nginx` and will run a single container using the `nginx:1.14.2` image. This container will listen on port 80.

On the other hand, the **Service** you've defined is creating a stable network interface to the nginx Pods created by the Deployment. The Service ensures that network traffic can reach one of the nginx Pods, even if the Pods are being moved around or replaced. Here's a breakdown of the Service:

- `apiVersion: v1`: This is the API version you're using to create the Service.
- `kind: Service`: This tells Kubernetes you're creating a Service.
- `metadata`: This section includes data about your Service, like its name.
- `spec`: This is where you define the behavior of the Service.
    - `type: NodePort`: This tells Kubernetes to make the Service accessible on a static port on each Node in the cluster.
    - `ports`: This is the list of ports the Service exposes. It's exposing port 80.
    - `selector`: This is how the Service identifies which Pods to direct network traffic to. In this case, it's looking for Pods with the label `app: nginx`.

So, to answer your question: the Deployment ensures that there are always three nginx Pods running, and the Service makes sure that those Pods can be reached over the network on a consistent IP address and port. You need both to expose your nginx application to network traffic reliably.

## so if i delete deployment,m then servicve will not work (i.e. the server is down)


Yes, that's correct. The Deployment in Kubernetes is responsible for creating and updating instances of your application (i.e., Pods). When you delete a Deployment, all the Pods managed by the Deployment will also be deleted, and thus the service will have no Pods to forward requests to.

The Service itself will still exist, but with no Pods matching its selector labels, it will not be able to forward any requests. In other words, your application (in this case, the server) would be considered "down" because there are no instances of it running for the Service to route traffic to.

Remember, a Service in Kubernetes is merely a stable interface to a set of Pods. It's the Deployment that's responsible for the actual lifecycle management of these Pods. Without the Pods, the Service has nothing to interface with.

So, in summary, if you delete the Deployment, the Service will have no functioning backends, and any requests made to the Service will fail.