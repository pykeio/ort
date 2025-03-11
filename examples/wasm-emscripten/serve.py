from http.server import HTTPServer, SimpleHTTPRequestHandler
from os import chdir, path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="serve debug build")
parser.add_argument("-r", "--release", action="store_true", help="serve release build")
args = parser.parse_args()
if args.release:
    mode = "release"
else:
    mode = "debug"

# SharedArrayBuffer required for threaded execution need specific CORS headers.
# See: https://rustwasm.github.io/wasm-bindgen/examples/raytrace.html#browser-requirements
# See: https://emscripten.org/docs/porting/pthreads.html
# See: https://stackoverflow.com/a/68358986
class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        SimpleHTTPRequestHandler.end_headers(self)

# Serve index.html.
chdir(path.join(path.dirname(__file__), f"target/wasm32-unknown-emscripten/{mode}"))
httpd = HTTPServer(("localhost", 5555), CORSRequestHandler)
print(f"Serving {mode} build at: http://localhost:5555")
httpd.serve_forever()
