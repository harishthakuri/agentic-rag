# Last-Modified header

The HTTP **`Last-Modified`** response header contains a date and time when the origin server believes the resource was last modified.
It is used as a validator in conditional requests (`If-Modified-Since` or `If-Unmodified-Since`) to determine if a requested resource is the same as one already stored by the client.
It is less accurate than an `ETag` for determining file contents, but can be used as a fallback mechanism if ETags are unavailable.

`Last-Modified` is also used by crawlers to adjust crawl frequency, by browsers in heuristic caching, and by content management systems (CMS) to display the time the content was last modified.

- Header type: Response header, Representation header
- Forbidden request header: No
- CORS-safelisted response header: Yes

## Syntax

```http
Last-Modified: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT
```

## Directives

- `<day-name>`
  - : One of "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", or "Sun" (case-sensitive).
- `<day>`
  - : 2 digit day number, e.g., "04" or "23".
- `<month>`
  - : One of "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" (case-sensitive).
- `<year>`
  - : 4 digit year number, e.g., "1990" or "2016".
- `<hour>`
  - : 2 digit hour number, e.g., "09" or "23".
- `<minute>`
  - : 2 digit minute number, e.g., "04" or "59".
- `<second>`
  - : 2 digit second number, e.g., "04" or "59".
- GMT
  - : Greenwich Mean Time. HTTP dates are always expressed in GMT, never in local time.

## Examples

```http
Last-Modified: Wed, 21 Oct 2015 07:28:00 GMT
```

## See also

- `Etag`
- HTTP Conditional Requests guide
- `If-Match`, `If-Modified-Since`, `If-Unmodified-Since`, `If-None-Match` conditional request headers
- `304 Not Modified`, `412 Precondition Failed` response status codes
