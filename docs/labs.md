# Labs

Experimental and in-progress notebooks live here. These are works-in-progress that explore new ideas, prototype features, or test approaches that haven't yet graduated into official tutorials. Expect rough edges.

<div class="labs-grid" markdown>
{% for card in labs %}
<div class="labs-card{% if card.url %} labs-card--linked{% endif %}" markdown>

![{{ card.title }}]({{ card.image }})

<div class="labs-card-body" markdown>

### {{ card.title }}

{{ card.description }}

<span class="labs-badge">{{ card.status }}</span>

{% if card.url %}<a class="labs-card-link" href="{{ card.url }}" target="_blank" rel="noopener"></a>{% endif %}

</div>
</div>
{% endfor %}
</div>
