# ETag header

The HTTP **`ETag`** (entity tag) response header is an identifier for a specific version of a resource.
It lets caches be more efficient and save bandwidth, as a web server does not need to resend a full response if the content has not changed.
Additionally, ETags help to prevent simultaneous updates of a resource from overwriting each other ("mid-air collisions").

If the resource at a given URL changes, a new `ETag` value _must_ be generated.
A comparison of them can determine whether two representations of a resource are the same.

- Header type: Response header, Representation header
- Forbidden request header: No

## Syntax

```http
ETag: W/"<etag_value>"
ETag: "<etag_value>"
```

## Directives

- `W/`
  - : `W/` (case-sensitive) indicates that a weak validator is used.
    Weak ETags are easy to generate, but are far less useful for comparisons.
    Strong validators are ideal for comparisons but can be very difficult to generate efficiently.
    Weak `ETag` values of two representations of the same resources might be semantically equivalent, but not byte-for-byte identical.
    This means weak ETags prevent caching when byte range requests are used, but strong ETags mean range requests can still be cached.
- `<etag_value>`
  - : Entity tag that uniquely represents the requested resource. It is a string of ASCII characters placed between double quotes, like `"675af34563dc-tr34"`.
    The method by which `ETag` values are generated is not specified.
    Typically, the ETag value is a hash of the content, a hash of the last modification timestamp, or just a revision number.
    For example, a wiki engine can use a hexadecimal hash of the documentation article content.

## Examples

```http
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
ETag: W/"0815"
```

### Avoiding mid-air collisions

With the help of the `ETag` and the `If-Match` headers, you can detect mid-air edit collisions (conflicts).

For example, when editing a wiki, the current wiki content may be hashed and put into an `ETag` header in the response:

```http
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
```

When saving changes to a wiki page (posting data), the `POST` request
will contain the `If-Match` header containing the `ETag`
values to check freshness against.

```http
If-Match: "33a64df551425fcc55e4d42a148795d9f25f89d4"
```

If the hashes don't match, it means that the document has been edited in-between and a
`412 Precondition Failed` error is thrown.

### Caching of unchanged resources

Another typical use of the `ETag` header is to cache resources that are unchanged.
If a user visits a given URL again (that has an `ETag` set), and it is _stale_ (too old to be considered usable), the client will send the value of its `ETag` along in an `If-None-Match` header field:

```http
If-None-Match: "33a64df551425fcc55e4d42a148795d9f25f89d4"
```

The server compares the client's `ETag` (sent with `If-None-Match`) with the `ETag` for its current version of the resource, and if both values match (that is, the resource has not changed), the server sends back a `304 Not Modified` status, without a body, which tells the client that the cached version of the response is still good to use (_fresh_).

## See also

- `If-Match`, `If-None-Match` headers
- `304 Not Modified`, `412 Precondition Failed` response status codes
- W3C Note: Editing the Web – Detecting the Lost Update Problem Using Unreserved Checkout
