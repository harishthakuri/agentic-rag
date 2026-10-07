# Vendored browser libraries

Served locally (not from a CDN) so the UI works offline and under a strict
Content-Security-Policy (`script-src 'self'`).

| File | Library | Version | License | Source |
|---|---|---|---|---|
| `marked-18.1.0.umd.js` | [marked](https://github.com/markedjs/marked) (Markdown → HTML) | 18.1.0 | MIT | `npm:marked@18.1.0/lib/marked.umd.js` |
| `purify-3.4.16.min.js` | [DOMPurify](https://github.com/cure53/DOMPurify) (HTML sanitiser) | 3.4.16 | Apache-2.0 / MPL-2.0 | `npm:dompurify@3.4.16/dist/purify.min.js` |

Model output is untrusted: it is always rendered as `DOMPurify.sanitize(marked.parse(text))`.
To upgrade, download the new files, update the names in `index.html` and this table.
