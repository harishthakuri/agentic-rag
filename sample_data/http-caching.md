# HTTP Caching

HTTP caching stores copies of responses so that later requests can be answered without contacting the origin server, or by transferring less data. Caches exist in browsers (private caches) and in proxies and CDNs (shared caches).

## Freshness

A cached response is either *fresh* or *stale*. A fresh response can be served directly. A stale one must usually be revalidated with the server first.

The `Cache-Control` response header controls freshness:

- `max-age=3600` makes the response fresh for 3600 seconds.
- `s-maxage=600` overrides `max-age` for shared caches only, such as CDNs.
- `no-cache` allows storing the response, but requires revalidation **before every reuse**. Despite its name, it does not mean "do not cache".
- `no-store` forbids storing the response anywhere. Use it for sensitive data.
- `private` allows only the browser to store the response, not shared caches. Use it for personalised pages.
- `public` allows shared caches to store the response, even for requests that carried an `Authorization` header.
- `immutable` tells the browser the response will never change while fresh, so it can skip revalidation even when the user reloads the page.

If a response has no explicit freshness information, caches may estimate a lifetime from the `Last-Modified` header. This is called heuristic freshness.

## Validation

When a cached response becomes stale, the cache can ask the server whether it is still valid, instead of downloading it again. This is a *conditional request*.

### ETag

An `ETag` is an opaque identifier for a specific version of a resource, often a hash of its content. The cache sends it back in `If-None-Match`:

```http
GET /app.js HTTP/1.1
If-None-Match: "33a64df5"
```

If the resource has not changed, the server replies `304 Not Modified` with no body, and the cache keeps using its copy. Otherwise it sends `200 OK` with the new content and a new ETag.

A weak validator, written `W/"33a64df5"`, means the two versions are semantically equivalent, though not necessarily byte-for-byte identical.

### Last-Modified

`Last-Modified` carries the time the resource last changed. The cache sends it back in `If-Modified-Since`. It only has one-second resolution, so ETags are more precise.

## Serving stale content

Two extensions let caches serve stale responses to improve speed and availability:

- `stale-while-revalidate=60` lets a cache serve a stale response for up to 60 seconds while it revalidates in the background. Users never wait for the revalidation.
- `stale-if-error=86400` lets a cache serve a stale response for up to a day when the origin returns an error or cannot be reached.

## The Vary header

A cache normally stores one response per URL. If the response depends on request headers, such as `Accept-Encoding` or `Accept-Language`, the server must list them in `Vary`:

```http
Vary: Accept-Encoding
```

The cache then stores a separate variant for each value. `Vary: *` effectively makes a response uncacheable. Varying on headers with many possible values, such as `User-Agent`, fragments the cache and lowers the hit rate.

## Cache busting

For static assets such as JavaScript and CSS, a common pattern is to put a content hash in the filename (`app.3f9a1c.js`), serve it with `Cache-Control: max-age=31536000, immutable`, and reference it from an HTML page that is always revalidated (`no-cache`). Deploying a new version changes the filename, so clients fetch it immediately, while unchanged assets stay cached for a year.
