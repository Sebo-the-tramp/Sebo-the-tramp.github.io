new Vue({
    el: '#app',
    data() {
        return {
            items: [
                {
                    action: '#',
                    title: 'C#',
                    active: true,
                    items: [
                        { title: 'Search' },
                        { title: 'Populate List Box' },
                        { title: 'Switch case' },
                    ]
                },
                {
                    action: 'web',
                    title: 'HTML',
                    items: [
                        { title: 'Breakfast & brunch' },
                        { title: 'New American' },
                        { title: 'Sushi' }
                    ]
                }
            ],
            cards: [
                { id: "c_Populate", text: "Cioa" }
            ]
        }
    }
})