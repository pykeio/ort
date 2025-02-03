from http.server import HTTPServer, SimpleHTTPRequestHandler
from os import chdir, path

# SharedArrayBuffer required for threaded execution need specific CORS headers.
# See: https://rustwasm.github.io/wasm-bindgen/examples/raytrace.html#browser-requirements
# See: https://emscripten.org/docs/porting/pthreads.html
# See: https://stackoverflow.com/a/68358986
class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        SimpleHTTPRequestHandler.end_headers(self)

# Serve index.html from release build.
chdir(path.join(path.dirname(__file__), "../../target/wasm32-unknown-emscripten/release"))
httpd = HTTPServer(("localhost", 5555), CORSRequestHandler)
print("Serving site at: http://localhost:5555")
httpd.serve_forever()