# If-Modified-Since header

The HTTP **`If-Modified-Since`** request header makes a request conditional.
The server sends back the requested resource, with a `200` status, only if it has been modified after the date in the `If-Modified-Since` header.
If the resource has not been modified since, the response is a `304` without any body, and the `Last-Modified` response header of the previous request contains the date of the last modification.

Unlike `If-Unmodified-Since`, `If-Modified-Since` can only be used with a `GET` or `HEAD`.
When used in combination with `If-None-Match`, it is ignored, unless the server doesn't support `If-None-Match`.

The most common use case is to update a cached entity that has no associated `ETag`.

- Header type: Request header
- Forbidden request header: No

## Syntax

```http
If-Modified-Since: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT
```

## Directives

- `<day-name>`
  - : One of `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, or `Sun` (case-sensitive).
- `<day>`
  - : 2 digit day number, e.g., "04" or "23".
- `<month>`
  - : One of `Jan`, `Feb`, `Mar`, `Apr`, `May`, `Jun`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec` (case sensitive).
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
If-Modified-Since: Wed, 21 Oct 2015 07:28:00 GMT
```

## See also

- `ETag`
- `If-Match`, `If-None-Match`, `If-Unmodified-Since` conditional request headers
- `304 Not Modified`, `412 Precondition Failed` response status codes
