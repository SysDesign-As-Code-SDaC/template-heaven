export function FeaturesSection() {
  const features = [
    {
      title: 'Fast Performance',
      description: 'Lightning-fast loading times and smooth user experience.',
      icon: '⚡'
    },
    {
      title: 'Secure & Reliable',
      description: 'Enterprise-grade security with 99.9% uptime guarantee.',
      icon: '🔒'
    },
    {
      title: 'Easy to Use',
      description: 'Intuitive interface designed for users of all skill levels.',
      icon: '🎯'
    },
    {
      title: 'Scalable',
      description: 'Grows with your business from startup to enterprise.',
      icon: '📈'
    },
    {
      title: '24/7 Support',
      description: 'Round-the-clock customer support when you need it.',
      icon: '🛟'
    },
    {
      title: 'Mobile Ready',
      description: 'Fully responsive design that works on all devices.',
      icon: '📱'
    }
  ]

  return (
    <section className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Why Choose Our Platform?
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            We've built our platform with the latest technologies and best practices
            to deliver an exceptional user experience.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow border border-gray-100"
            >
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
