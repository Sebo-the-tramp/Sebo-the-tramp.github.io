obj = new Vue({
    el: '#app',
    data() {
        return {
            window: {
                width: 0,
                height: 0
            }
        }
    },
    created() {
        window.addEventListener('resize', this.handleResize)
        window.location.replace("www.sebastiancavada.it");
        this.handleResize();        
    },
    destroyed() {
        window.removeEventListener('resize', this.handleResize)
    },
    methods: {
        getHeight() {
            console.log(window.innerHeight)
            this.window.width = window.innerWidth;
            this.window.height = window.innerHeight;
            console.log(this.window.width, this.window.height)
            return window.innerHeight;
        }
    }
})
