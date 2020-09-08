default: static/bundle.js static/bundle.css static/fonts

clean:
	rm -f static/bundle.js static/bundle.css
	rm -rf static/fonts

static/bundle.js: client.js html_construction.js package-lock.json
	./node_modules/.bin/browserify -d -t [ babelify --presets [ env ] ] ./client.js > $@

static/bundle.css: client.less package-lock.json
	./node_modules/.bin/lessc ./client.less $@

static/fonts:
	cp -r node_modules/font-awesome/fonts static/fonts
