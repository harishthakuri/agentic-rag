# Kubernetes Services and Ingress

Pods are ephemeral: they are created and destroyed, and each gets a new IP address. A **Service** gives a changing set of pods one stable network identity. An **Ingress** exposes HTTP services to the outside world.

## How a Service selects pods

A Service uses a label selector to choose its backend pods. The control plane keeps an EndpointSlice up to date with the IP addresses of the pods that match the selector **and** are ready. A pod that fails its readiness probe is removed from the endpoints, so it receives no traffic until it is ready again.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: api
spec:
  selector:
    app: api
  ports:
    - port: 80
      targetPort: 8080
```

`port` is the port the Service listens on. `targetPort` is the port on the pods that traffic is forwarded to.

## Service types

### ClusterIP

ClusterIP is the default type. The Service gets a virtual IP address that is reachable only from inside the cluster. Use it for internal communication between components.

### NodePort

A NodePort Service opens the same port on every node, by default in the range 30000–32767. Traffic to `<any node IP>:<nodePort>` is forwarded to the Service. NodePort builds on ClusterIP: a NodePort Service also has a cluster IP.

### LoadBalancer

A LoadBalancer Service asks the cloud provider to create an external load balancer that forwards to the Service. It builds on NodePort. On bare-metal clusters, an add-on such as MetalLB is needed to provide the external IP addresses.

### ExternalName

An ExternalName Service has no selector and no proxying. It returns a DNS CNAME record pointing to an external hostname, such as a managed database.

### Headless Services

Setting `clusterIP: None` creates a headless Service. It has no virtual IP; DNS returns the IP addresses of the individual pods instead. StatefulSets use headless Services so that each pod gets a stable DNS name such as `db-0.db.default.svc.cluster.local`.

## Service discovery with DNS

The cluster DNS server (usually CoreDNS) creates a record for every Service:

```
<service>.<namespace>.svc.cluster.local
```

Pods in the same namespace can simply use the Service name, for example `http://api`.

## kube-proxy

kube-proxy runs on every node and implements the virtual IPs. In `iptables` mode it writes NAT rules that pick a backend pod at random for each new connection. In `ipvs` mode it uses the kernel's IP Virtual Server, which scales better to thousands of Services and supports more load-balancing algorithms. Some network plugins, such as Cilium, replace kube-proxy with eBPF programs.

## Ingress

An Ingress defines HTTP and HTTPS routing rules from outside the cluster to Services, based on hostnames and URL paths. A single external entry point can serve many Services:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web
spec:
  tls:
    - hosts: [shop.example.com]
      secretName: shop-tls
  rules:
    - host: shop.example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api
                port:
                  number: 80
```

An Ingress resource does nothing by itself: an **Ingress controller**, such as ingress-nginx or Traefik, must be running in the cluster to implement the rules. TLS is terminated at the controller using the certificate stored in the referenced Secret.

The newer **Gateway API** is the successor to Ingress. It separates infrastructure concerns (Gateway) from routing rules (HTTPRoute) and supports more protocols and traffic-splitting features.
