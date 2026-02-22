# Labs

Experimental and in-progress notebooks live here. These are works-in-progress that explore new ideas, prototype features, or test approaches that haven't yet graduated into official tutorials. Expect rough edges.

<div class="labs-grid" markdown>
{% for card in labs %}
<div class="labs-card" markdown>

![{{ card.title }}]({{ card.image }})

<div class="labs-card-body" markdown>

### {{ card.title }}

{{ card.description }}

<span class="labs-badge">{{ card.status }}</span>

</div>
</div>
{% endfor %}
</div>
